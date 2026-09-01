#!/usr/bin/env python3
"""
Stage A of the copy-paste pipeline: run SAM3 ONCE over a sample of train
instances and cache each product cutout (feathered BGRA sticker) to disk, so
Stage B (generate_copy_paste_dataset.py) can compose thousands of synthetic
scenes without re-running SAM3 for every draw.

Sampling is category-stratified (round-robin over the ~200 RPC train
categories via the COCO train annotations) rather than a flat random shuffle,
so every product category ends up represented in the sticker cache instead of
risking rare categories being skipped entirely.

Usage (server):
  python tools/extract_stickers.py --n-stickers 50000

Note: RPC train2019 has ~53.7k single-product images total, so 50k stickers
means near-full coverage of the split (~250/category); the actual saved count
may land a bit lower if SAM3 fails/skips some instances.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from copy_paste_demo import extract_cutout
from sam3_segment import best_mask_for_box, load_yolo_boxes, yolo_to_xyxy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `import src.*` when run as tools/extract_stickers.py

from src.config import get_project_root, load_config
from src.converter import build_image_annotation_map, build_image_info_map, load_coco_annotations


def build_category_map(cfg, split: str) -> dict:
    """file_name -> category_id, from the original COCO train annotations (train
    images are single-product, so the first/only annotation's category is used)."""
    project_root = get_project_root()
    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    ann_path = dataset_root / cfg.dataset.annotations[split]

    coco = load_coco_annotations(ann_path)
    img_info = build_image_info_map(coco)
    ann_map = build_image_annotation_map(coco)
    category_map = {}
    for img_id, info in img_info.items():
        anns = ann_map.get(img_id, [])
        if anns:
            category_map[info["file_name"]] = anns[0]["category_id"]
    return category_map


def iter_all_instances(dataset_root: Path, split: str, seed: int, category_map: dict = None):
    """Yield every (image_path, box) pair across the split. Boxes of the same
    image are always yielded consecutively so the caller can re-use one SAM3
    image encoding per image.

    If category_map is given, images are visited in category-stratified
    round-robin order (one image per category per round) instead of a flat
    shuffle, so an early stop at n_stickers still covers every category."""
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    rng = random.Random(seed)

    if category_map:
        by_category = {}
        uncategorized = []
        for img_path in image_paths:
            cat_id = category_map.get(img_path.name)
            (by_category.setdefault(cat_id, []) if cat_id is not None else uncategorized).append(img_path)
        for bucket in by_category.values():
            rng.shuffle(bucket)
        rng.shuffle(uncategorized)
        buckets = list(by_category.values()) + ([uncategorized] if uncategorized else [])
        rng.shuffle(buckets)
        ordered = []
        while any(buckets):
            for bucket in buckets:
                if bucket:
                    ordered.append(bucket.pop())
            buckets = [b for b in buckets if b]
        image_paths = ordered
    else:
        rng.shuffle(image_paths)

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = load_yolo_boxes(label_path)
        for box in boxes:
            yield img_path, box


def load_existing_index(output_dir: Path) -> tuple[list, set]:
    """Resume support: reload index.json, drop entries whose PNG is missing/corrupt
    (freeing their disk space), and return (valid_index, {already-used source stems})."""
    index_path = output_dir / "index.json"
    if not index_path.exists():
        return [], set()

    with open(index_path) as f:
        raw_index = json.load(f)

    valid_index = []
    used_sources = set()
    dropped = 0
    for entry in raw_index:
        img_path = output_dir / entry["file"]
        bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED) if img_path.exists() else None
        if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
            img_path.unlink(missing_ok=True)  # corrupt/partial write (e.g. disk was full) — free the space
            dropped += 1
            continue
        valid_index.append(entry)
        used_sources.add(entry["source"])

    print(f"Resuming: {len(valid_index)} valid stickers already cached, {dropped} corrupt entries dropped")
    return valid_index, used_sources


def main():
    parser = argparse.ArgumentParser(description="Stage A: cache SAM3 product cutouts as reusable stickers")
    parser.add_argument("--dataset", default="yolo_dataset_rpc")
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-stickers", type=int, default=50000)
    parser.add_argument("--output", default="sticker_cache")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="config.yaml", help="used only to locate the COCO train annotations for category-stratified sampling")
    parser.add_argument("--no-stratify", action="store_true", help="fall back to a flat random shuffle instead of per-category round-robin")
    args = parser.parse_args()

    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        # sam3 hardcodes device="cuda" in several tensor-creation calls regardless
        # of the requested device; redirect those calls to cpu wherever they show up
        def _cpu_safe(fn):
            def wrapper(*a, **kw):
                if kw.get("device") == "cuda":
                    kw["device"] = "cpu"
                return fn(*a, **kw)
            return wrapper
        for _name in ("zeros", "ones", "empty", "full", "arange", "tensor", "rand", "randn", "eye", "linspace"):
            setattr(torch, _name, _cpu_safe(getattr(torch, _name)))

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    print(f"Loading SAM3 from checkpoint: {args.checkpoint}")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, device=args.device)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.confidence)
    print(f"Model ready. Device: {args.device}\n")

    dataset_root = Path(args.dataset).expanduser()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    category_map = None
    if not args.no_stratify:
        try:
            cfg = load_config(args.config)
            category_map = build_category_map(cfg, args.split)
            print(f"Category-stratified sampling over {len(set(category_map.values()))} categories "
                  f"({len(category_map)} images mapped)")
        except Exception as e:
            print(f"WARNING: could not build category map ({e}), falling back to flat shuffle")

    _geo_keys = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
    index, used_sources = load_existing_index(output_dir)
    saved = len(index)
    tried = 0
    current_img_path = None
    current_image = None
    state = None

    pbar = tqdm(total=args.n_stickers, initial=saved, desc="Extracting stickers", unit="sticker")
    for img_path, (cx, cy, bw, bh) in iter_all_instances(dataset_root, args.split, args.seed, category_map):
        if saved >= args.n_stickers:
            break
        if img_path.stem in used_sources:
            continue  # already extracted in a previous (resumed) run
        tried += 1
        pbar.set_postfix(tried=tried)

        if img_path != current_img_path:
            try:
                current_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                tqdm.write(f"  skip {img_path.name}: cannot open ({e})")
                current_img_path = None
                continue
            try:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    state = processor.set_image(current_image)
            except Exception as e:
                tqdm.write(f"  skip {img_path.name}: encode failed ({e})")
                current_img_path = None
                continue
            current_img_path = img_path
        else:
            for key in _geo_keys:
                state.pop(key, None)

        W, H = current_image.size
        box_xyxy = yolo_to_xyxy(cx, cy, bw, bh, W, H)

        try:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                output = processor.add_geometric_prompt(box=[cx, cy, bw, bh], label=True, state=state)
            sam3_masks = output.get("masks")
            sam3_boxes = output.get("boxes")
            mask = None
            if sam3_masks is not None and sam3_masks.numel() > 0 and sam3_boxes is not None:
                mask = best_mask_for_box(box_xyxy, sam3_masks, sam3_boxes.float())
        except Exception as e:
            tqdm.write(f"  skip {img_path.name}: SAM3 failed ({e})")
            continue

        if mask is None:
            continue

        result = extract_cutout(current_image, mask)
        if result is None:
            continue

        bgr, alpha = result
        if bgr.shape[0] < 12 or bgr.shape[1] < 12:
            continue  # too tiny to be a useful sticker

        alpha_u8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        bgra = np.dstack([bgr, alpha_u8])
        sticker_name = f"{img_path.stem}_{saved:06d}.png"
        out_path = output_dir / sticker_name
        if not cv2.imwrite(str(out_path), bgra):
            out_path.unlink(missing_ok=True)
            pbar.close()
            print(f"\nERROR: cv2.imwrite failed for {out_path} (libpng \"Write Error\" usually means the disk "
                  f"is full). Stopping here so no more GPU time is wasted.")
            print("Free up disk space (check `df -h`) and re-run the exact same command — "
                  "already-cached stickers are skipped automatically (resume).")
            with open(output_dir / "index.json", "w") as f:
                json.dump(index, f)
            return
        category_id = category_map.get(img_path.name) if category_map else None
        index.append({"file": sticker_name, "source": img_path.stem, "category_id": category_id})
        used_sources.add(img_path.stem)
        saved += 1
        pbar.update(1)
        if saved % 500 == 0:
            with open(output_dir / "index.json", "w") as f:
                json.dump(index, f)

    pbar.close()

    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f)

    print(f"\nDone. {saved} stickers saved / {tried} instances tried -> {output_dir}")


if __name__ == "__main__":
    main()
