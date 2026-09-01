#!/usr/bin/env python3
"""
Stage B of the copy-paste pipeline: compose synthetic multi-product training
scenes from cached stickers (see extract_stickers.py), using empirical
instance-count / bbox-size distributions extracted by analyze_dataset_stats.py
(val_stats.json) so synthetic scenes match real checkout-scene statistics.
Writes new images + YOLO labels directly into the training split (never
val/test) so they can be trained on immediately, no SAM3 needed at this stage.

Usage (server):
  python extract_stickers.py --n-stickers 6000                                # once
  python analyze_dataset_stats.py --split val                                  # once
  python generate_copy_paste_dataset.py --n-images 50000 --purge-originals     # repeatable

--purge-originals removes the old single-product symlinked images/labels (i.e.
any images/train or labels/train file NOT matching the synthetic --prefix) so
the train split ends up purely synthetic multi-product scenes.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from copy_paste_demo import grid_positions, paste_object, resize_cutout, resolve_visible_boxes


def load_sticker_paths(sticker_dir: Path) -> list[Path]:
    with open(sticker_dir / "index.json") as f:
        index = json.load(f)
    return [sticker_dir / entry["file"] for entry in index]


def load_sticker_image(path: Path):
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        return None
    bgr = bgra[:, :, :3]
    alpha = bgra[:, :, 3].astype(np.float32) / 255.0
    return bgr, alpha


def make_background(canvas_size: int, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    """Slightly textured, per-image randomized neutral canvas so the model can't
    learn a fixed flat-background shortcut."""
    base = rng.randint(170, 210)
    tint = np.array([base + rng.randint(-8, 8) for _ in range(3)], dtype=np.float32)
    noise = np_rng.normal(0, 4, size=(canvas_size, canvas_size, 3))
    canvas = np.clip(tint[None, None, :] + noise, 0, 255).astype(np.uint8)
    return canvas


def purge_original_images(images_out: Path, labels_out: Path, prefix: str) -> int:
    """Delete every images/train + labels/train file NOT produced by this script
    (i.e. the original single-product symlinks from convert.py), keyed off the
    synthetic filename prefix."""
    removed = 0
    for img_path in list(images_out.iterdir()):
        if img_path.name.startswith(f"{prefix}_"):
            continue
        img_path.unlink()
        label_path = labels_out / f"{img_path.stem}.txt"
        if label_path.exists():
            label_path.unlink()
        removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description="Stage B: generate synthetic copy-paste training scenes")
    parser.add_argument("--dataset", default="yolo_dataset_rpc")
    parser.add_argument("--sticker-dir", default="sticker_cache")
    parser.add_argument("--stats", default="val_stats.json")
    parser.add_argument("--n-images", type=int, default=50000)
    parser.add_argument("--canvas-size", type=int, default=1280)
    parser.add_argument("--visibility-thresh", type=float, default=0.3)
    parser.add_argument("--min-instances", type=int, default=3)
    parser.add_argument("--max-instances", type=int, default=20)
    parser.add_argument("--prefix", default="synpaste")
    parser.add_argument("--preview-every", type=int, default=200)
    parser.add_argument("--preview-dir", default="_copy_paste_preview")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--purge-originals", action="store_true",
                         help="remove the old single-product images/labels from train (keep only synthetic)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    with open(args.stats) as f:
        stats = json.load(f)
    instance_samples = stats["instances_per_image"]["samples"]
    width_ratio_samples = stats["bbox_width_ratio"]["samples"]

    sticker_dir = Path(args.sticker_dir)
    sticker_paths = load_sticker_paths(sticker_dir)
    if len(sticker_paths) < 2:
        print(f"Not enough stickers found in {sticker_dir}, run tools/extract_stickers.py first.")
        return
    print(f"Loaded {len(sticker_paths)} cached stickers from {sticker_dir}")

    dataset_root = Path(args.dataset).expanduser()
    images_out = dataset_root / "images" / "train"
    labels_out = dataset_root / "labels" / "train"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_dir)

    if args.purge_originals:
        removed = purge_original_images(images_out, labels_out, args.prefix)
        print(f"Purged {removed} original single-product images/labels from {images_out}")

    written = 0
    for i in tqdm(range(args.n_images), desc="Generating synthetic scenes", unit="img"):
        n_instances = int(np.clip(rng.choice(instance_samples), args.min_instances, args.max_instances))
        chosen_paths = rng.choices(sticker_paths, k=n_instances)

        cutouts = [c for c in (load_sticker_image(p) for p in chosen_paths) if c is not None]
        if len(cutouts) < 2:
            continue

        canvas = make_background(args.canvas_size, rng, np_rng)
        positions = grid_positions(len(cutouts), args.canvas_size,
                                    cell_margin=args.canvas_size // 10, rng=rng)

        footprints = []
        for (bgr, alpha), (cx, cy) in zip(cutouts, positions):
            w_ratio = rng.choice(width_ratio_samples)
            target_w = max(16, int(w_ratio * args.canvas_size))
            bgr_r, alpha_r = resize_cutout(bgr, alpha, target_w)
            footprint, _ = paste_object(canvas, bgr_r, alpha_r, cx, cy)
            footprints.append(footprint)

        visible_boxes = resolve_visible_boxes(footprints, args.visibility_thresh)
        H, W = canvas.shape[:2]
        yolo_lines = []
        for box in visible_boxes:
            if box is None:
                continue
            x1, y1, x2, y2 = box
            cx_n, cy_n = ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H
            w_n, h_n = (x2 - x1) / W, (y2 - y1) / H
            yolo_lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")

        if not yolo_lines:
            continue

        name = f"{args.prefix}_{i:06d}"
        cv2.imwrite(str(images_out / f"{name}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (labels_out / f"{name}.txt").write_text("\n".join(yolo_lines) + "\n")
        written += 1

        if args.preview_every > 0 and written % args.preview_every == 1:
            preview_dir.mkdir(parents=True, exist_ok=True)
            boxed = canvas.copy()
            for box in visible_boxes:
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                cv2.rectangle(boxed, (x1, y1), (x2, y2), (50, 205, 50), 2)
            cv2.imwrite(str(preview_dir / f"{name}_preview.jpg"), boxed)

    print(f"\nDone. {written}/{args.n_images} synthetic images written -> {images_out}")
    print(f"Preview samples -> {preview_dir}")


if __name__ == "__main__":
    main()
