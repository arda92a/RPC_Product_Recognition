#!/usr/bin/env python3
"""
Teacher inference -> pseudo-labels on the UNLABELED real checkout pool
(val2019_unlabeled, produced by tools/split_val_pool.py).

Runs the current best (synthetic-trained) detector on val2019_unlabeled once
at a permissive confidence floor, then lets you sweep the actual keep-
threshold against the real GT (QA only) before committing anything to disk —
important because a threshold that's too high can leave most real instances
unlabeled, and YOLO's loss then treats those un-boxed regions as background,
actively teaching the Student to suppress recall on exactly the hard cases
we're trying to fix.

Writes YOLO-format pseudo labels into <dataset-root>/images/train_pseudo +
labels/train_pseudo — meant to be combined with the existing (purely
synthetic) train split via a separate dataset_uda.yaml with
    train: [images/train, images/train_pseudo]
so the pure-synthetic train split is never touched/overwritten.

If instances_val2019_unlabeled.json (the real GT, saved by split_val_pool.py
for QA only) is available, this script reports the pseudo-labels'
precision/recall against it — purely diagnostic, never used to influence
training or filtering.

IMPORTANT: never point --images-dir at val2019_clean or test2019 — those are
reserved for early-stopping / final evaluation and must never be trained on.

Usage:
  # 1) sweep thresholds first, writes nothing to disk:
  python tools/generate_pseudo_labels.py --detector-checkpoint runs/.../best.pt \
      --conf-sweep 0.85 0.7 0.6 0.5 0.4 0.3

  # 2) once you've picked a threshold, commit it to disk:
  python tools/generate_pseudo_labels.py --detector-checkpoint runs/.../best.pt --conf 0.5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src.*` when run as tools/generate_pseudo_labels.py

from src.config import get_project_root, load_config
from src.converter import load_coco_annotations
from src.detection_evaluator import _coco_to_xyxy, _match_boxes

RawPreds = Dict[str, dict]  # file_name -> {"boxes": [N,4] xyxy, "scores": [N], "h": int, "w": int}


def load_area_ratio_bounds(stats_path: Path, lo_pct: str = "p1", hi_pct: str = "p99") -> Optional[tuple]:
    if not stats_path.exists():
        return None
    with open(stats_path, "r") as f:
        stats = json.load(f)
    s = stats["bbox_area_ratio"]["stats"]
    return s[lo_pct], s[hi_pct]


def run_teacher_inference(detector_path: str, images_dir: Path, conf_floor: float, iou: float,
                           imgsz: int, device: str, max_images: Optional[int] = None) -> RawPreds:
    """Single inference pass at a permissive confidence floor — every threshold in
    --conf-sweep (all >= conf_floor) is then just a post-hoc filter over these boxes."""
    image_paths = sorted(images_dir.glob("*.jpg"))
    if max_images is not None:
        image_paths = image_paths[:max_images]

    detector = YOLO(detector_path)
    raw: RawPreds = {}
    for img_path in tqdm(image_paths, desc="teacher inference", unit="img"):
        result = detector.predict(source=str(img_path), conf=conf_floor, iou=iou, imgsz=imgsz,
                                   device=device, verbose=False)[0]
        h, w = result.orig_shape
        boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4), dtype=np.float32)
        scores = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty((0,), dtype=np.float32)
        raw[img_path.name] = {"boxes": boxes, "scores": scores, "h": h, "w": w}
    return raw


def select_boxes(entry: dict, threshold: float, area_ratio_bounds: Optional[tuple]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Filter one image's raw boxes by confidence threshold + box-size sanity. Returns (kept_boxes_xyxy, kept_scores, n_dropped_by_size)."""
    boxes, scores, h, w = entry["boxes"], entry["scores"], entry["h"], entry["w"]
    conf_mask = scores >= threshold
    boxes, scores = boxes[conf_mask], scores[conf_mask]

    if area_ratio_bounds is None or len(boxes) == 0:
        return boxes, scores, 0

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (w * h)
    size_mask = (areas >= area_ratio_bounds[0]) & (areas <= area_ratio_bounds[1])
    return boxes[size_mask], scores[size_mask], int((~size_mask).sum())


def load_gt_by_file(unlabeled_ann_path: Path) -> Dict[str, List[list]]:
    if not unlabeled_ann_path.exists():
        return {}
    coco = load_coco_annotations(unlabeled_ann_path)
    images = {img["id"]: img for img in coco["images"]}
    gt_by_file: Dict[str, List[list]] = {}
    for ann in coco["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            continue
        gt_by_file.setdefault(img["file_name"], []).append(_coco_to_xyxy(ann["bbox"]))
    return gt_by_file


def evaluate_threshold(raw: RawPreds, threshold: float, area_ratio_bounds: Optional[tuple],
                        gt_by_file: Dict[str, List[list]], iou_threshold: float = 0.5) -> dict:
    """QA-only: precision/recall of pseudo-labels at this threshold against real GT. No disk writes."""
    images_with_boxes, boxes_kept, boxes_dropped_size = 0, 0, 0
    tp_total, fp_total, fn_total = 0, 0, 0

    for file_name, entry in raw.items():
        kept_boxes, kept_scores, dropped = select_boxes(entry, threshold, area_ratio_bounds)
        boxes_kept += len(kept_boxes)
        boxes_dropped_size += dropped
        if len(kept_boxes) > 0:
            images_with_boxes += 1

        gt_boxes = gt_by_file.get(file_name)
        if gt_boxes is None:
            continue
        gt_arr = np.array(gt_boxes, dtype=np.float32)
        tp, fp, fn = _match_boxes(kept_boxes.astype(np.float32), kept_scores.astype(np.float32), gt_arr, iou_threshold)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    return {
        "threshold": threshold, "images_with_boxes": images_with_boxes, "boxes_kept": boxes_kept,
        "boxes_dropped_size": boxes_dropped_size, "precision": precision, "recall": recall,
        "tp": tp_total, "fp": fp_total, "fn": fn_total,
    }


def write_pseudo_labels(raw: RawPreds, threshold: float, area_ratio_bounds: Optional[tuple],
                         images_dir: Path, out_images_dir: Path, out_labels_dir: Path) -> None:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for file_name, entry in raw.items():
        kept_boxes, _, _ = select_boxes(entry, threshold, area_ratio_bounds)
        if len(kept_boxes) == 0:
            continue
        h, w = entry["h"], entry["w"]
        lines = []
        for x1, y1, x2, y2 in kept_boxes:
            xc, yc = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        img_path = images_dir / file_name
        dst_img = out_images_dir / file_name
        if not dst_img.exists():
            dst_img.symlink_to(img_path.resolve())
        (out_labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1

    print(f"\nWrote pseudo-labels for {written}/{len(raw)} images -> {out_images_dir} / {out_labels_dir}")


def print_sweep_table(rows: List[dict]) -> None:
    cols = ["threshold", "images_with_boxes", "boxes_kept", "boxes_dropped_size", "precision", "recall", "tp", "fp", "fn"]
    print("\n" + " | ".join(f"{c:>12}" for c in cols))
    print("-" * (15 * len(cols)))
    for row in sorted(rows, key=lambda r: -r["threshold"]):
        cells = [f"{row['threshold']:.2f}", row["images_with_boxes"], row["boxes_kept"], row["boxes_dropped_size"],
                 f"{row['precision']:.4f}", f"{row['recall']:.4f}", row["tp"], row["fp"], row["fn"]]
        print(" | ".join(f"{c:>12}" for c in cells))


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
    parser.add_argument("--conf", type=float, default=0.85, help="Keep-threshold used to WRITE pseudo-labels (ignored if --conf-sweep given)")
    parser.add_argument("--conf-sweep", type=float, nargs="+", default=None,
                         help="Dry-run: report precision/recall at each threshold against real GT, write nothing to disk")
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
    conf_floor = min(args.conf_sweep) if args.conf_sweep else args.conf

    print(f"Detector      : {detector_path}")
    print(f"Images        : {images_dir}")
    print(f"conf_floor={conf_floor} iou={args.iou} imgsz={args.imgsz}")
    print(f"Area-ratio filter bounds: {area_ratio_bounds}")
    print()

    raw = run_teacher_inference(detector_path, images_dir, conf_floor, args.iou, args.imgsz, args.device,
                                 max_images=args.max_images)
    gt_by_file = load_gt_by_file(unlabeled_ann_path)
    if not gt_by_file:
        print(f"(No QA: {unlabeled_ann_path} not found)")

    if args.conf_sweep:
        rows = [evaluate_threshold(raw, t, area_ratio_bounds, gt_by_file, args.iou) for t in args.conf_sweep]
        print_sweep_table(rows)
        print("\nDry run only — nothing written. Re-run with --conf <chosen threshold> (no --conf-sweep) to commit to disk.")
        return

    stats = evaluate_threshold(raw, args.conf, area_ratio_bounds, gt_by_file, args.iou)
    print(f"Images with pseudo-lbl: {stats['images_with_boxes']}")
    print(f"Boxes kept            : {stats['boxes_kept']}")
    print(f"Boxes dropped (size)  : {stats['boxes_dropped_size']}")
    if gt_by_file:
        print("\n=== QA vs. real GT (diagnostic only, not used for filtering/training) ===")
        print(f"pseudo-label precision: {stats['precision']:.4f}   recall: {stats['recall']:.4f}   "
              f"(TP={stats['tp']} FP={stats['fp']} FN={stats['fn']})")

    write_pseudo_labels(raw, args.conf, area_ratio_bounds, images_dir, out_images_dir, out_labels_dir)


if __name__ == "__main__":
    main()
