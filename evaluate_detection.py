#!/usr/bin/env python3
"""
Pure product-detection evaluation on the RPC test set: measures the detector's
recall/precision/instance-counting accuracy in isolation from the metric-
learning classifier, per clutter level (easy/medium/hard). Also runs the
standard YOLO mAP evaluation (src/evaluator.py) for the box-quality picture.

Useful for diagnosing whether a poor full-pipeline cAcc (see
evaluate_pipeline.py) stems from missed/duplicate detections rather than
misclassification.

Usage:
  python evaluate_detection.py
  python evaluate_detection.py --detector-checkpoint runs/rpc_singleclass_yolo11l/weights/best.pt --conf 0.25 --iou 0.5
"""

import argparse

from src.config import get_project_root, load_config
from src.detection_evaluator import evaluate_detector_counting
from src.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser(description="Detection-only evaluation on the RPC test set")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--detector-checkpoint", default=None, help="Default: cfg.evaluation.model")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--iou-match-threshold", type=float, default=0.5,
                         help="IoU threshold for greedy pred<->GT box matching")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of test images (debugging)")
    parser.add_argument("--skip-map", action="store_true", help="Skip the standard ultralytics mAP evaluation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = get_project_root()
    detector_checkpoint = args.detector_checkpoint or str(project_root / cfg.evaluation.model)

    if not args.skip_map:
        print("=== Standard YOLO mAP evaluation (yolo_dataset/test) ===")
        evaluate(cfg, model_path=detector_checkpoint)
        print()

    print("=== Class-agnostic recall/precision/counting analysis (RPC test annotations) ===")
    evaluate_detector_counting(cfg, detector_checkpoint, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                               iou_match_threshold=args.iou_match_threshold, max_images=args.max_images)


if __name__ == "__main__":
    main()
