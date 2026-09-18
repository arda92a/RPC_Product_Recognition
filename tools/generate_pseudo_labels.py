#!/usr/bin/env python3
"""
Teacher inference -> pseudo-labels on the UNLABELED real checkout pool
(val2019_unlabeled, produced by tools/split_val_pool.py).

Runs the current best (synthetic-trained) detector on val2019_unlabeled with a
high confidence threshold, drops boxes whose size is wildly inconsistent with
the real bbox-area-ratio distribution (val_stats.json), and writes YOLO-format
pseudo labels into <dataset-root>/images/train_pseudo + labels/train_pseudo.

This directory is meant to be combined with the existing (purely synthetic)
train split via a separate dataset_uda.yaml with
    train: [images/train, images/train_pseudo]
so the pure-synthetic train split is never touched/overwritten.

If instances_val2019_unlabeled.json (the real GT, saved by split_val_pool.py
for QA only) is available, this script also reports the pseudo-labels'
precision/recall against it — purely diagnostic, never used to influence
training or filtering.

IMPORTANT: never point --images-dir at val2019_clean or test2019 — those are
reserved for early-stopping / final evaluation and must never be trained on.

Usage:
  python tools/generate_pseudo_labels.py --detector-checkpoint runs/.../best.pt
  python tools/generate_pseudo_labels.py --conf 0.85 --max-images 200   # quick QA run
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src.*` when run as tools/generate_pseudo_labels.py

from src.config import get_project_root, load_config
from src.converter import load_coco_annotations
from src.detection_evaluator import _coco_to_xyxy, _match_boxes


def load_area_ratio_bounds(stats_path: Path, lo_pct: str = "p1", hi_pct: str = "p99") -> Optional[tuple]:
    if not stats_path.exists():
        return None
    with open(stats_path, "r") as f:
        stats = json.load(f)
    s = stats["bbox_area_ratio"]["stats"]
    return s[lo_pct], s[hi_pct]


def generate_pseudo_labels(
    detector_path: str,
    images_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    area_ratio_bounds: Optional[tuple],
    max_images: Optional[int] = None,
) -> Dict[str, dict]:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    detector = YOLO(detector_path)

    manifest: Dict[str, dict] = {}
    total_boxes_kept, total_boxes_dropped_size, images_with_boxes = 0, 0, 0

    for img_path in tqdm(image_paths, desc="pseudo-labeling", unit="img"):
        result = detector.predict(source=str(img_path), conf=conf, iou=iou, imgsz=imgsz,
                                   device=device, verbose=False)[0]
        h, w = result.orig_shape

        boxes_xyxy = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4))
        scores = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty((0,))

        lines = []
        kept, dropped = 0, 0
        for (x1, y1, x2, y2), score in zip(boxes_xyxy, scores):
            area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
            if area_ratio_bounds is not None and not (area_ratio_bounds[0] <= area_ratio <= area_ratio_bounds[1]):
                dropped += 1
                continue
            xc, yc = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            kept += 1

        if kept == 0:
            continue  # skip images with no surviving pseudo-boxes — nothing useful to train on

        dst_img = out_images_dir / img_path.name
        if not dst_img.exists():
            dst_img.symlink_to(img_path.resolve())
        (out_labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")

        manifest[img_path.name] = {"kept": kept, "dropped_size": dropped, "mean_conf": float(scores.mean()) if len(scores) else 0.0}
        total_boxes_kept += kept
        total_boxes_dropped_size += dropped
        images_with_boxes += 1

    print(f"\nImages processed      : {len(image_paths)}")
    print(f"Images with pseudo-lbl: {images_with_boxes}")
    print(f"Boxes kept            : {total_boxes_kept}")
    print(f"Boxes dropped (size)  : {total_boxes_dropped_size}")
    return manifest


def qa_against_real_gt(unlabeled_ann_path: Path, out_labels_dir: Path, images_dir: Path,
                        iou_threshold: float = 0.5) -> None:
    """Diagnostic-only: compare pseudo-labels against the real GT saved by split_val_pool.py.
    Never used to filter/influence the pseudo-labels themselves."""
    if not unlabeled_ann_path.exists():
        print(f"\n(No QA: {unlabeled_ann_path} not found)")
        return

    coco = load_coco_annotations(unlabeled_ann_path)
    images = {img["id"]: img for img in coco["images"]}
    images_by_file = {img["file_name"]: img for img in coco["images"]}
    gt_by_file: Dict[str, List[list]] = {}
    for ann in coco["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            continue
        gt_by_file.setdefault(img["file_name"], []).append(_coco_to_xyxy(ann["bbox"]))

    tp_total, fp_total, fn_total = 0, 0, 0
    for file_name, gt_boxes in gt_by_file.items():
        label_path = out_labels_dir / f"{Path(file_name).stem}.txt"
        pred_boxes = []
        if label_path.exists():
            img = images_by_file[file_name]
            W, H = img["width"], img["height"]
            for line in label_path.read_text().splitlines():
                _, xc, yc, bw, bh = (float(v) for v in line.split())
                pred_boxes.append([(xc - bw / 2) * W, (yc - bh / 2) * H, (xc + bw / 2) * W, (yc + bh / 2) * H])
        pred_arr = np.array(pred_boxes, dtype=np.float32) if pred_boxes else np.empty((0, 4), dtype=np.float32)
        gt_arr = np.array(gt_boxes, dtype=np.float32)
        scores = np.ones(len(pred_arr), dtype=np.float32)
        tp, fp, fn = _match_boxes(pred_arr, scores, gt_arr, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    print("\n=== QA vs. real GT (diagnostic only, not used for filtering/training) ===")
    print(f"pseudo-label precision: {precision:.4f}   recall: {recall:.4f}   (TP={tp_total} FP={fp_total} FN={fn_total})")


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-labels on val2019_unlabeled for Teacher-Student self-training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--detector-checkpoint", default=None, help="Default: cfg.evaluation.model")
    parser.add_argument("--images-dir", default=None,
                         help="Default: <project_root>/val2019_unlabeled (from tools/split_val_pool.py)")
    parser.add_argument("--unlabeled-annotations", default=None,
                         help="Default: <project_root>/instances_val2019_unlabeled.json (QA only)")
    parser.add_argument("--dataset-root", default="yolo_dataset_rpc", help="Where images/train_pseudo + labels/train_pseudo are written")
    parser.add_argument("--stats", default="val_stats.json", help="Box-size sanity filter reference (from tools/analyze_dataset_stats.py)")
    parser.add_argument("--conf", type=float, default=0.85)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images (debugging)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = get_project_root()

    detector_path = args.detector_checkpoint or str(project_root / cfg.evaluation.model)
    images_dir = Path(args.images_dir) if args.images_dir else project_root / "val2019_unlabeled"
    unlabeled_ann_path = Path(args.unlabeled_annotations) if args.unlabeled_annotations \
        else project_root / "instances_val2019_unlabeled.json"

    forbidden_dirs = {(project_root / "val2019_clean").resolve()}
    dataset_root_cfg = Path(cfg.dataset.root)
    if dataset_root_cfg.is_absolute():
        forbidden_dirs.add((dataset_root_cfg / cfg.dataset.images["test"]).resolve())
    if images_dir.resolve() in forbidden_dirs:
        raise SystemExit(f"Refusing to run: {images_dir} is a held-out eval split, not the unlabeled pool.")

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    out_images_dir = dataset_root / "images" / "train_pseudo"
    out_labels_dir = dataset_root / "labels" / "train_pseudo"

    area_ratio_bounds = load_area_ratio_bounds(project_root / args.stats)

    print(f"Detector      : {detector_path}")
    print(f"Images        : {images_dir}")
    print(f"Output        : {out_images_dir} / {out_labels_dir}")
    print(f"conf={args.conf} iou={args.iou} imgsz={args.imgsz}")
    print(f"Area-ratio filter bounds: {area_ratio_bounds}")
    print()

    generate_pseudo_labels(detector_path, images_dir, out_images_dir, out_labels_dir,
                            args.conf, args.iou, args.imgsz, args.device,
                            area_ratio_bounds, max_images=args.max_images)

    qa_against_real_gt(unlabeled_ann_path, out_labels_dir, images_dir)


if __name__ == "__main__":
    main()
