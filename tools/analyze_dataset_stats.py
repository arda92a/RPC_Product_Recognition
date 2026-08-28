#!/usr/bin/env python3
"""
Extract real-world statistics from an RPC COCO split (val by default) to drive
the SAM3 copy-paste synthetic scene generator:
  - instances-per-image distribution (how many products per checkout scene)
  - bbox width/height ratio-to-image-size distributions (how big products appear)
  - bbox area ratio and aspect ratio distributions
  - category frequency (reference only, detector is single-class)

Saves raw empirical samples + summary percentiles to a JSON file so the
generator can sample directly from real distributions instead of guessing.

Usage:
  python tools/analyze_dataset_stats.py --split val
  python tools/analyze_dataset_stats.py --split test --output test_stats.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src.*` when run as tools/analyze_dataset_stats.py

from src.config import get_project_root, load_config
from src.converter import build_image_annotation_map, build_image_info_map, load_coco_annotations


def percentiles(values, ps=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    result = {f"p{p}": float(np.percentile(arr, p)) for p in ps}
    result.update(min=float(arr.min()), max=float(arr.max()),
                  mean=float(arr.mean()), std=float(arr.std()), count=int(arr.size))
    return result


def ascii_hist(values, bins=12, width=40):
    if not values:
        return ""
    arr = np.array(values, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=bins)
    max_c = max(int(counts.max()), 1)
    lines = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(c / max_c * width)
        lines.append(f"  [{lo:8.3f}, {hi:8.3f}) {c:6d} {bar}")
    return "\n".join(lines)


def analyze(cfg, split: str) -> dict:
    project_root = get_project_root()
    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    ann_file = dataset_root / cfg.dataset.annotations[split]

    coco = load_coco_annotations(ann_file)
    img_ann_map = build_image_annotation_map(coco)
    img_info_map = build_image_info_map(coco)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    instances_per_image = []
    bbox_w_ratio, bbox_h_ratio, bbox_area_ratio, bbox_aspect = [], [], [], []
    category_counts = Counter()

    for image_id, info in img_info_map.items():
        anns = img_ann_map.get(image_id, [])
        W, H = info["width"], info["height"]
        instances_per_image.append(len(anns))
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            bbox_w_ratio.append(w / W)
            bbox_h_ratio.append(h / H)
            bbox_area_ratio.append((w * h) / (W * H))
            bbox_aspect.append(w / h)
            category_counts[cat_names.get(ann["category_id"], str(ann["category_id"]))] += 1

    return {
        "split": split,
        "num_images": len(img_info_map),
        "num_annotations": sum(len(v) for v in img_ann_map.values()),
        "instances_per_image": {"samples": instances_per_image, "stats": percentiles(instances_per_image)},
        "bbox_width_ratio": {"samples": bbox_w_ratio, "stats": percentiles(bbox_w_ratio)},
        "bbox_height_ratio": {"samples": bbox_h_ratio, "stats": percentiles(bbox_h_ratio)},
        "bbox_area_ratio": {"samples": bbox_area_ratio, "stats": percentiles(bbox_area_ratio)},
        "bbox_aspect_ratio": {"samples": bbox_aspect, "stats": percentiles(bbox_aspect)},
        "category_frequency": dict(category_counts.most_common()),
    }


def print_summary(stats: dict):
    print(f"\n=== {stats['split']} split ===")
    print(f"images: {stats['num_images']}  annotations: {stats['num_annotations']}")

    ipi = stats["instances_per_image"]["stats"]
    print("\nInstances per image:")
    print(f"  mean={ipi['mean']:.2f} std={ipi['std']:.2f} min={ipi['min']:.0f} "
          f"p50={ipi['p50']:.0f} p90={ipi['p90']:.0f} p99={ipi['p99']:.0f} max={ipi['max']:.0f}")
    print(ascii_hist(stats["instances_per_image"]["samples"]))

    for key, label in [
        ("bbox_width_ratio", "bbox width / image width"),
        ("bbox_height_ratio", "bbox height / image height"),
        ("bbox_area_ratio", "bbox area / image area"),
        ("bbox_aspect_ratio", "bbox aspect ratio (w/h)"),
    ]:
        s = stats[key]["stats"]
        print(f"\n{label}:")
        print(f"  mean={s['mean']:.4f} p10={s['p10']:.4f} p50={s['p50']:.4f} "
              f"p90={s['p90']:.4f} p99={s['p99']:.4f} max={s['max']:.4f}")

    print("\nTop 10 categories by frequency:")
    for name, count in list(stats["category_frequency"].items())[:10]:
        print(f"  {name}: {count}")
    print("Bottom 5 categories by frequency:")
    for name, count in list(stats["category_frequency"].items())[-5:]:
        print(f"  {name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Extract RPC split statistics for copy-paste generation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="Output JSON path (default: <split>_stats.json)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stats = analyze(cfg, args.split)
    print_summary(stats)

    output_path = Path(args.output) if args.output else Path(f"{args.split}_stats.json")
    with open(output_path, "w") as f:
        json.dump(stats, f)
    print(f"\nSaved full stats (with raw samples for empirical sampling) -> {output_path}")


if __name__ == "__main__":
    main()
