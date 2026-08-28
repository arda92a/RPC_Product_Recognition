#!/usr/bin/env python3
"""
Stage A of the copy-paste pipeline: run SAM3 ONCE over a sample of train
instances and cache each product cutout (feathered BGRA sticker) to disk, so
Stage B (generate_copy_paste_dataset.py) can compose thousands of synthetic
scenes without re-running SAM3 for every draw.

Usage (server):
  python extract_stickers.py --n-stickers 6000
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from copy_paste_demo import extract_cutout
from sam3_segment import best_mask_for_box, load_yolo_boxes, yolo_to_xyxy


def iter_all_instances(dataset_root: Path, split: str, seed: int):
    """Yield every (image_path, box) pair across the split, image order shuffled.
    Boxes of the same image are always yielded consecutively so the caller can
    re-use one SAM3 image encoding per image."""
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(image_paths)
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = load_yolo_boxes(label_path)
        for box in boxes:
            yield img_path, box


def main():
    parser = argparse.ArgumentParser(description="Stage A: cache SAM3 product cutouts as reusable stickers")
    parser.add_argument("--dataset", default="yolo_dataset_rpc")
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-stickers", type=int, default=6000)
    parser.add_argument("--output", default="sticker_cache")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
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

    _geo_keys = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
    index = []
    saved = 0
    tried = 0
    current_img_path = None
    current_image = None
    state = None

    pbar = tqdm(total=args.n_stickers, desc="Extracting stickers", unit="sticker")
    for img_path, (cx, cy, bw, bh) in iter_all_instances(dataset_root, args.split, args.seed):
        if saved >= args.n_stickers:
            break
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
        cv2.imwrite(str(output_dir / sticker_name), bgra)
        index.append({"file": sticker_name, "source": img_path.stem})
        saved += 1
        pbar.update(1)

    pbar.close()

    with open(output_dir / "index.json", "w") as f:
        json.dump(index, f)

    print(f"\nDone. {saved} stickers saved / {tried} instances tried -> {output_dir}")


if __name__ == "__main__":
    main()
