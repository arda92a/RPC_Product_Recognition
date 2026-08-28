#!/usr/bin/env python3
"""
Retrieval evaluation for the trained metric-learning model: build a gallery
from one split and query it with another, reporting top-1/top-5 accuracy + mAP.

Usage:
  python evaluate_metric.py --gallery-split train --query-split test
"""

import argparse

from src.config import get_project_root, load_config
from src.metric.evaluator import evaluate_metric_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate the metric-learning model via retrieval")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (default: <output_dir>/best.pt)")
    parser.add_argument("--gallery-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--query-split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = str(get_project_root() / cfg.metric_training.output_dir / "best.pt")

    evaluate_metric_model(cfg, checkpoint, gallery_split=args.gallery_split, query_split=args.query_split)


if __name__ == "__main__":
    main()
