"""
Retrieval-based evaluation: build an embedding gallery from one split
(typically train, clean studio crops) and query it with another split
(val/test, real checkout crops), reporting top-1/top-5 accuracy and mAP —
the metric that actually matters for deployed product recognition.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import Config, get_project_root
from src.metric.backbone import DinoV3Backbone
from src.metric.dataset import build_metric_datasets
from src.metric.model import MetricModel


@torch.no_grad()
def extract_embeddings(model: MetricModel, loader: DataLoader, device):
    model.eval()
    all_embeds, all_labels = [], []
    for images, labels in loader:
        embeds = model(images.to(device))
        all_embeds.append(embeds.cpu())
        all_labels.append(labels)
    return torch.cat(all_embeds), torch.cat(all_labels)


def retrieval_metrics(gallery_emb, gallery_labels, query_emb, query_labels, ks=(1, 5)):
    sims = query_emb @ gallery_emb.T  # embeddings are L2-normalized -> cosine similarity
    ranked = sims.argsort(dim=1, descending=True)

    results = {}
    for k in ks:
        top_k = ranked[:, :k]
        correct = (gallery_labels[top_k] == query_labels.unsqueeze(1)).any(dim=1)
        results[f"top{k}"] = correct.float().mean().item()

    aps = []
    for i in range(query_emb.size(0)):
        matches = (gallery_labels[ranked[i]] == query_labels[i]).float()
        if matches.sum() == 0:
            aps.append(0.0)
            continue
        cum_hits = matches.cumsum(dim=0)
        precision_at_k = cum_hits / torch.arange(1, matches.size(0) + 1, dtype=torch.float32)
        aps.append(((precision_at_k * matches).sum() / matches.sum()).item())
    results["mAP"] = sum(aps) / len(aps)
    return results


def evaluate_metric_model(cfg: Config, checkpoint_path: str,
                           gallery_split: str = "train", query_split: str = "test"):
    mc = cfg.metric_training
    project_root = get_project_root()
    dataset_root = Path(mc.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    use_cuda = mc.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{mc.device}" if use_cuda else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone_checkpoint = None
    if mc.backbone_checkpoint:
        path = Path(mc.backbone_checkpoint)
        backbone_checkpoint = str(path if path.is_absolute() else project_root / path)
    backbone = DinoV3Backbone(mc.backbone, source=mc.backbone_source, hf_model_id=mc.hf_model_id,
                               checkpoint_path=backbone_checkpoint)
    model = MetricModel(backbone, embed_dim=mc.embed_dim, hidden_dim=mc.hidden_dim,
                         dropout=mc.dropout).to(device)
    model.backbone.load_state_dict(checkpoint["backbone"])
    model.head.load_state_dict(checkpoint["head"])

    train_ds, val_ds, test_ds = build_metric_datasets(dataset_root, mc.img_size)
    splits = {"train": train_ds, "val": val_ds, "test": test_ds}
    gallery_ds, query_ds = splits[gallery_split], splits[query_split]

    gallery_loader = DataLoader(gallery_ds, batch_size=mc.batch_size, shuffle=False, num_workers=mc.num_workers)
    query_loader = DataLoader(query_ds, batch_size=mc.batch_size, shuffle=False, num_workers=mc.num_workers)

    print(f"Extracting gallery embeddings ({gallery_split}, {len(gallery_ds)} crops)...")
    gallery_emb, gallery_labels = extract_embeddings(model, gallery_loader, device)
    print(f"Extracting query embeddings ({query_split}, {len(query_ds)} crops)...")
    query_emb, query_labels = extract_embeddings(model, query_loader, device)

    metrics = retrieval_metrics(gallery_emb, gallery_labels, query_emb, query_labels)
    print("\n--- Retrieval Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return metrics
