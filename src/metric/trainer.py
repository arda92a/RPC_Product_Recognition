"""
Staged metric-learning trainer.

  Stage 1 (linear probe): backbone fully frozen, train only the embedding
    head + margin-loss class weights. Fast, stabilizes the head before
    touching pretrained DINOv3 features.
  Stage 2 (partial fine-tune): unfreeze the last N backbone blocks with a
    much lower (discriminative) learning rate, keep training head + loss
    weights at the stage-1 rate.
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, get_project_root
from src.metric.backbone import DinoV3Backbone
from src.metric.dataset import build_metric_datasets
from src.metric.losses import build_loss
from src.metric.model import MetricModel


def _build_optimizer(head_params, backbone_params, lr_head, lr_backbone, weight_decay):
    groups = [{"params": head_params, "lr": lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _resolve_checkpoint_path(checkpoint: str, project_root: Path):
    if not checkpoint:
        return None
    path = Path(checkpoint)
    return str(path if path.is_absolute() else project_root / path)


def _init_wandb(cfg: Config, num_classes: int):
    wc = cfg.wandb
    if not wc.enabled:
        return None
    try:
        import wandb

        mc = cfg.metric_training
        run = wandb.init(
            project=wc.project,
            name=f"{wc.run_name}-metric",
            config={
                "backbone": mc.backbone,
                "loss": mc.loss,
                "margin": mc.margin,
                "scale": mc.scale,
                "embed_dim": mc.embed_dim,
                "img_size": mc.img_size,
                "batch_size": mc.batch_size,
                "stage1_epochs": mc.stage1_epochs,
                "stage1_lr_head": mc.stage1_lr_head,
                "stage2_epochs": mc.stage2_epochs,
                "stage2_lr_head": mc.stage2_lr_head,
                "stage2_lr_backbone": mc.stage2_lr_backbone,
                "unfreeze_last_n_blocks": mc.unfreeze_last_n_blocks,
                "num_classes": num_classes,
            },
        )
        print(f"W&B run      : {run.url}\n")
        return run
    except ImportError:
        print("WARNING: wandb not installed — skipping W&B logging. Run: pip install wandb\n")
        return None


def train_metric_model(cfg: Config) -> str:
    mc = cfg.metric_training
    project_root = get_project_root()
    dataset_root = Path(mc.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    use_cuda = mc.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{mc.device}" if use_cuda else "cpu")

    train_ds, val_ds, _ = build_metric_datasets(dataset_root, mc.img_size)
    num_classes = len(train_ds.classes)
    print(f"Classes: {num_classes}  Train crops: {len(train_ds)}  Val crops: {len(val_ds)}")
    print(f"Device: {device}\n")

    wandb_run = _init_wandb(cfg, num_classes)

    train_loader = DataLoader(train_ds, batch_size=mc.batch_size, shuffle=True,
                               num_workers=mc.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=mc.batch_size, shuffle=False,
                             num_workers=mc.num_workers, pin_memory=True)

    checkpoint_path = _resolve_checkpoint_path(mc.backbone_checkpoint, project_root)
    backbone = DinoV3Backbone(mc.backbone, source=mc.backbone_source, hf_model_id=mc.hf_model_id,
                               checkpoint_path=checkpoint_path)
    model = MetricModel(backbone, embed_dim=mc.embed_dim, hidden_dim=mc.hidden_dim,
                         dropout=mc.dropout).to(device)
    loss_fn = build_loss(mc.loss, mc.embed_dim, num_classes, mc.margin, mc.scale, mc.subcenters).to(device)

    output_dir = project_root / mc.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_val_acc = 0.0

    print("=== Stage 1: linear probe (backbone frozen) ===")
    model.backbone.freeze()
    optimizer = _build_optimizer(
        list(model.head.parameters()) + list(loss_fn.parameters()),
        [], mc.stage1_lr_head, 0.0, mc.weight_decay,
    )
    best_val_acc = _run_epochs(model, loss_fn, train_loader, val_loader, optimizer,
                                mc.stage1_epochs, device, best_val_acc, best_path, "stage1",
                                wandb_run=wandb_run, step_offset=0)

    if mc.stage2_epochs > 0:
        print(f"\n=== Stage 2: fine-tune last {mc.unfreeze_last_n_blocks} backbone blocks ===")
        model.backbone.unfreeze_last_n_blocks(mc.unfreeze_last_n_blocks)
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        optimizer = _build_optimizer(
            list(model.head.parameters()) + list(loss_fn.parameters()),
            backbone_params, mc.stage2_lr_head, mc.stage2_lr_backbone, mc.weight_decay,
        )
        best_val_acc = _run_epochs(model, loss_fn, train_loader, val_loader, optimizer,
                                    mc.stage2_epochs, device, best_val_acc, best_path, "stage2",
                                    wandb_run=wandb_run, step_offset=mc.stage1_epochs)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")

    if wandb_run is not None:
        import wandb
        wandb_run.summary["best_val_acc"] = best_val_acc
        wandb.finish()

    return str(best_path)


def _run_epochs(model, loss_fn, train_loader, val_loader, optimizer, epochs,
                 device, best_val_acc, best_path, stage, wandb_run=None, step_offset=0):
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = _run_one_epoch(model, loss_fn, train_loader, device, optimizer,
                                                desc=f"[{stage}] epoch {epoch}/{epochs} train")
        val_loss, val_acc = _run_one_epoch(model, loss_fn, val_loader, device, optimizer=None,
                                            desc=f"[{stage}] epoch {epoch}/{epochs} val")

        print(f"[{stage}] epoch {epoch}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.1f}s)")

        if wandb_run is not None:
            global_step = step_offset + epoch
            wandb_run.log({
                f"{stage}/train_loss": train_loss,
                f"{stage}/train_acc": train_acc,
                f"{stage}/val_loss": val_loss,
                f"{stage}/val_acc": val_acc,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "epoch": global_step,
            }, step=global_step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "backbone": model.backbone.state_dict(),
                "head": model.head.state_dict(),
                "loss_fn": loss_fn.state_dict(),
                "classes": train_loader.dataset.classes,
            }, best_path)

    return best_val_acc


def _run_one_epoch(model, loss_fn, loader, device, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn.train(is_train)

    running_loss, running_correct, running_total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            loss, logits = loss_fn(embeddings, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += images.size(0)
            pbar.set_postfix(loss=running_loss / running_total, acc=running_correct / running_total)

    return running_loss / running_total, running_correct / running_total
