"""
Retrieval-based evaluation: build an embedding gallery from one split
(typically train, clean studio crops) and query it with another split
(val/test, real checkout crops), reporting top-1/top-5 accuracy and mAP —
the metric that actually matters for deployed product recognition.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, get_project_root
from src.metric.backbone import DinoV3Backbone
from src.metric.dataset import build_metric_datasets
from src.metric.model import MetricModel


@torch.no_grad()
def extract_embeddings(model: MetricModel, loader: DataLoader, device, desc: str = ""):
    model.eval()
    all_embeds, all_labels = [], []
    for images, labels in tqdm(loader, desc=desc, unit="batch"):
        embeds = model(images.to(device))
        all_embeds.append(embeds.cpu())
        all_labels.append(labels)
    return torch.cat(all_embeds), torch.cat(all_labels)


def retrieval_metrics(gallery_emb, gallery_labels, query_emb, query_labels, ks=(1, 5),
                       device=None, chunk_size: int = 1024):
    """Chunked cosine-similarity retrieval: computing the full [Nq, Ng] similarity
    matrix at once (e.g. 185k x 322k floats = ~223GB) blows up memory, so queries
    are processed chunk_size at a time against the whole gallery instead."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gallery_emb = gallery_emb.to(device)
    gallery_labels = gallery_labels.to(device)

    top_k_hits = {k: 0 for k in ks}
    ap_sum = 0.0
    n_queries = query_emb.size(0)

    for start in tqdm(range(0, n_queries, chunk_size), desc="retrieval", unit="chunk"):
        end = min(start + chunk_size, n_queries)
        q_chunk = query_emb[start:end].to(device)
        q_labels_chunk = query_labels[start:end].to(device)

        sims = q_chunk @ gallery_emb.T  # [chunk, Ng], embeddings are L2-normalized -> cosine similarity
        ranked = sims.argsort(dim=1, descending=True)
        matches = (gallery_labels[ranked] == q_labels_chunk.unsqueeze(1)).float()  # [chunk, Ng]

        for k in ks:
            correct = matches[:, :k].any(dim=1)
            top_k_hits[k] += correct.sum().item()

        cum_hits = matches.cumsum(dim=1)
        ranks = torch.arange(1, matches.size(1) + 1, device=device, dtype=torch.float32)
        total_matches = matches.sum(dim=1).clamp(min=1)
        ap = ((cum_hits / ranks) * matches).sum(dim=1) / total_matches
        ap_sum += ap.sum().item()

        del sims, ranked, matches, cum_hits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = {f"top{k}": top_k_hits[k] / n_queries for k in ks}
    results["mAP"] = ap_sum / n_queries
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
    gallery_emb, gallery_labels = extract_embeddings(model, gallery_loader, device, desc="gallery")
    print(f"Extracting query embeddings ({query_split}, {len(query_ds)} crops)...")
    query_emb, query_labels = extract_embeddings(model, query_loader, device, desc="query")

    metrics = retrieval_metrics(gallery_emb, gallery_labels, query_emb, query_labels, device=device)
    print("\n--- Retrieval Evaluation ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if cfg.wandb.enabled:
        try:
            import wandb

            wc = cfg.wandb
            run = wandb.init(project=wc.project, name=f"{wc.run_name}-metric-eval",
                              config={"gallery_split": gallery_split, "query_split": query_split})
            run.log({f"retrieval/{k}": v for k, v in metrics.items()})
            wandb.finish()
        except ImportError:
            print("WARNING: wandb not installed — skipping W&B logging. Run: pip install wandb")

    return metrics
