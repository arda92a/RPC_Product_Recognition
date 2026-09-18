#!/usr/bin/env python3
"""
One-time split of val2019 (6,000 images) into two disjoint, non-overlapping
pools for the Teacher-Student / pseudo-labeling experiment:

  - val2019_clean     : labeled subset — the ONLY validation signal used during
                         Student fine-tuning (early stopping / best.pt
                         selection). Student never trains on these images.
  - val2019_unlabeled : the other subset — used strictly as an UNLABELED image
                         pool for pseudo-label generation. Its real COCO
                         annotations are saved to a side JSON for offline
                         pseudo-label QA only (never fed to the training
                         data loader).

Default split is 50/50 (3,000 / 3,000) — a larger clean val set gives a more
statistically stable early-stopping signal (this project has already been
bitten twice by patience/early-stopping instability), at the cost of a
smaller pseudo-label pool.

test2019 (24k images) is never touched by this script — it stays 100% locked
for the final cAcc/mCIoU/recall evaluation, per the agreed plan.

Usage:
  python tools/split_val_pool.py
  python tools/split_val_pool.py --clean-fraction 0.5 --copy
"""

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src.*` when run as tools/split_val_pool.py

from src.config import get_project_root, load_config
from src.converter import load_coco_annotations


def split_image_ids(coco_data: dict, clean_fraction: float, seed: int):
    """Stratify by RPC's 'level' field (easy/medium/hard) when present, else plain shuffle."""
    images = coco_data["images"]
    by_level = defaultdict(list)
    for img in images:
        by_level[img.get("level", "_none")].append(img["id"])

    rng = random.Random(seed)
    clean_ids, unlabeled_ids = set(), set()
    for level, ids in by_level.items():
        ids = ids[:]
        rng.shuffle(ids)
        n_clean = max(1, round(len(ids) * clean_fraction))
        clean_ids.update(ids[:n_clean])
        unlabeled_ids.update(ids[n_clean:])
    return clean_ids, unlabeled_ids


def build_subset_coco(coco_data: dict, image_ids: set) -> dict:
    images = [img for img in coco_data["images"] if img["id"] in image_ids]
    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in image_ids]
    return {
        "images": images,
        "annotations": annotations,
        "categories": coco_data["categories"],
    }


def link_images(image_dir: Path, file_names, out_dir: Path, use_symlinks: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    for file_name in file_names:
        src = image_dir / file_name
        dst = out_dir / file_name
        if dst.exists() or not src.exists():
            continue
        if use_symlinks:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Split val2019 into a clean early-stopping set and an unlabeled pseudo-label pool")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--val-annotations", default=None,
                         help="Override path to the val COCO json (default: dataset.root/annotations.val, "
                              "or ./instances_val.json if present in the project root)")
    parser.add_argument("--clean-fraction", type=float, default=0.5,
                         help="Fraction of val2019 kept as the labeled early-stopping set (default 0.5 -> 3000/6000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args()

    cfg = load_config(args.config)
    project_root = get_project_root()
    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    local_override = project_root / "instances_val.json"
    if args.val_annotations:
        val_ann_path = Path(args.val_annotations)
        output_dir = project_root
    elif local_override.exists():
        val_ann_path = local_override
        output_dir = project_root
    else:
        val_ann_path = dataset_root / cfg.dataset.annotations["val"]
        output_dir = dataset_root
    val_img_dir = dataset_root / cfg.dataset.images["val"]

    print(f"Loading val annotations from {val_ann_path}")
    coco_data = load_coco_annotations(val_ann_path)

    clean_ids, unlabeled_ids = split_image_ids(coco_data, args.clean_fraction, args.seed)
    print(f"val2019 total     : {len(coco_data['images'])}")
    print(f"val2019_clean     : {len(clean_ids)}  (labeled, used for Student early-stopping only)")
    print(f"val2019_unlabeled : {len(unlabeled_ids)}  (treated as UNLABELED for pseudo-label generation)")

    clean_coco = build_subset_coco(coco_data, clean_ids)
    unlabeled_coco = build_subset_coco(coco_data, unlabeled_ids)

    clean_ann_path = output_dir / "instances_val2019_clean.json"
    unlabeled_ann_path = output_dir / "instances_val2019_unlabeled.json"  # QA-only, never used for training
    clean_ann_path.write_text(json.dumps(clean_coco))
    unlabeled_ann_path.write_text(json.dumps(unlabeled_coco))
    print(f"\nWrote {clean_ann_path}")
    print(f"Wrote {unlabeled_ann_path}  (offline pseudo-label QA only — do NOT feed to training)")

    # images are only ever read from dataset_root (often a read-only mount) — the
    # linked-image output dirs must live under output_dir, which is guaranteed writable
    clean_img_dir = output_dir / "val2019_clean"
    unlabeled_img_dir = output_dir / "val2019_unlabeled"
    if val_img_dir.exists():
        link_images(val_img_dir, [img["file_name"] for img in clean_coco["images"]], clean_img_dir, not args.copy)
        link_images(val_img_dir, [img["file_name"] for img in unlabeled_coco["images"]], unlabeled_img_dir, not args.copy)
        print(f"Linked images into {clean_img_dir} and {unlabeled_img_dir}")
    else:
        print(f"\nNOTE: image dir {val_img_dir} not found on this machine — annotation JSONs were "
              f"still written, but no images were linked. Re-run this script on the machine that "
              f"actually has val2019/ (e.g. asusgpu) to produce {clean_img_dir.name}/{unlabeled_img_dir.name}.")

    print("\nNext steps:")
    print("  1. Point config.yaml's dataset.images.val -> 'val2019_clean' and")
    print("     dataset.annotations.val -> 'instances_val2019_clean.json', then re-run convert.py")
    print("     so YOLO training's early-stopping signal comes only from this clean, untouched subset.")
    print("  2. Run pseudo-label generation against 'val2019_unlabeled' (images only, ignore its GT).")
    print("  3. test2019 stays untouched throughout — final evaluation only.")


if __name__ == "__main__":
    main()
