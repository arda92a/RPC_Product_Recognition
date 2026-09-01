#!/usr/bin/env python3
"""
Full ACO (Automatic Checkout) pipeline evaluation: YOLO product detector +
metric-learning classifier (1-NN retrieval against metric_dataset/train),
scored on the RPC test set with the official RPC metrics — cAcc, ACD, mCCD,
mCIoU — per clutter level (easy/medium/hard), as defined in the RPC paper
(rpc_paper/) and implemented by rpctool
(https://github.com/DIYer22/retail_product_checkout_tools).

Usage:
  python evaluate_pipeline.py
  python evaluate_pipeline.py --detector-checkpoint runs/rpc_singleclass_yolo11l/weights/best.pt \
      --metric-checkpoint runs/metric/best.pt --conf 0.25 --iou 0.5
"""

import argparse

from src.config import get_project_root, load_config
from src.pipeline_evaluator import evaluate_pipeline


def main():
    parser = argparse.ArgumentParser(description="Full detection + recognition ACO pipeline evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--detector-checkpoint", default=None, help="Default: cfg.evaluation.model")
    parser.add_argument("--metric-checkpoint", default=None, help="Default: <metric_training.output_dir>/best.pt")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of test images (debugging)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = get_project_root()

    detector_checkpoint = args.detector_checkpoint or str(project_root / cfg.evaluation.model)
    metric_checkpoint = args.metric_checkpoint or str(project_root / cfg.metric_training.output_dir / "best.pt")

    evaluate_pipeline(cfg, detector_checkpoint, metric_checkpoint,
                       conf=args.conf, iou=args.iou, imgsz=args.imgsz, max_images=args.max_images)


if __name__ == "__main__":
    main()
