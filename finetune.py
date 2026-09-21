#!/usr/bin/env python3
"""
Student fine-tuning on synthetic + pseudo-labeled real data.

Loads a pre-trained detector (typically from pure synthetic training) and
fine-tunes it on the combined dataset_uda.yaml (train + train_pseudo splits)
at a much lower learning rate and fewer epochs, using early-stopping against
the clean val split (val2019_clean) only.

This bridges the domain gap from synthetic → real checkout images via
pseudo-labels, while keeping the validation signal completely held-out and
untouched (no data leakage).

Usage:
  python finetune.py --pretrained-checkpoint runs/rpc_singleclass_yolo11l_50k_lr_experiment/weights/best.pt \
      --epochs 30 --lr0 0.0003 --batch 16 --name rpc_student_uda_finetune
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import Config, get_project_root, load_config


def finetune(cfg: Config, pretrained_checkpoint: str, dataset_yaml: str = None,
              epochs: int = 30, lr0: float = 0.0003, batch: int = 16,
              device: str = "0", name: str = "rpc_student_uda_finetune") -> str:
    """Fine-tune a pre-trained detector on the UDA dataset (synthetic + pseudo-labeled real)."""
    project_root = get_project_root()
    wc = cfg.wandb

    if dataset_yaml is None:
        dataset_yaml = str(project_root / "dataset_uda.yaml")

    print(f"Pre-trained model: {pretrained_checkpoint}")
    print(f"Dataset YAML      : {dataset_yaml}")
    print(f"Epochs            : {epochs}")
    print(f"LR0               : {lr0}")
    print(f"Batch size        : {batch}")
    print(f"Device            : {device}")
    print()

    model = YOLO(pretrained_checkpoint)

    # Load model weights from checkpoint (warm-start, not pretrained backbone)
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=cfg.training.imgsz,
        batch=batch,
        device=device,
        workers=cfg.training.workers,
        optimizer=cfg.training.optimizer,
        lr0=lr0,
        lrf=cfg.training.lrf,
        patience=cfg.training.patience,
        project=str(project_root / cfg.training.project),
        name=name,
        pretrained=False,  # already have weights from checkpoint, don't reset
        resume=False,
        exist_ok=False,
        mosaic=cfg.training.mosaic,
        mixup=cfg.training.mixup,
        copy_paste=cfg.training.copy_paste,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nFine-tuning complete. Best weights: {best_weights}")

    # Evaluate on test split
    print("\n--- Running test-set evaluation ---")
    from src.evaluator import evaluate

    metrics = evaluate(cfg, model_path=str(best_weights))

    if wc.enabled:
        import wandb

        wandb_run = wandb.init(
            project=wc.project,
            name=f"{name}-eval",
            config={"epochs": epochs, "lr0": lr0, "batch": batch},
        )
        wandb_run.summary.update({
            "test/map50": metrics["map50"],
            "test/map50_95": metrics["map50_95"],
            "test/precision": metrics["precision"],
            "test/recall": metrics["recall"],
        })
        wandb.finish()

    return str(best_weights)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on synthetic + pseudo-labeled real data (UDA)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--pretrained-checkpoint", required=True,
                         help="Path to pre-trained model (e.g. runs/.../best.pt)")
    parser.add_argument("--dataset-yaml", type=str, default=None,
                         help="Override dataset.yaml path (default: dataset_uda.yaml)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr0", type=float, default=0.0003, help="Initial learning rate")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--name", type=str, default="rpc_student_uda_finetune")
    args = parser.parse_args()

    cfg = load_config(args.config)
    finetune(cfg, args.pretrained_checkpoint, dataset_yaml=args.dataset_yaml,
              epochs=args.epochs, lr0=args.lr0, batch=args.batch,
              device=args.device, name=args.name)


if __name__ == "__main__":
    main()
