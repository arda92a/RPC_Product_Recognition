#!/usr/bin/env python3
"""
One-shot proof-of-concept for SAM3-based copy-paste augmentation.

Picks N random product instances from different train images, segments each
with SAM3, cuts it out (feathered alpha), and composites all of them onto a
single background scene. Saves a clean composite + a version with recomputed
(occlusion-aware) bounding boxes so we can eyeball the result before building
the full batch pipeline.

Usage (server):
  python copy_paste_demo.py --n-products 7
  python copy_paste_demo.py --n-products 7 --background some_shelf.jpg
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3_segment import load_yolo_boxes, yolo_to_xyxy, best_mask_for_box


# ---------------------------------------------------------------------------
# Instance picking
# ---------------------------------------------------------------------------

def iter_candidate_instances(dataset_root: Path, split: str, seed: int):
    """Yield (image_path, (cx, cy, bw, bh)) — one random box per image, shuffled."""
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    rng = random.Random(seed)
    rng.shuffle(image_paths)
    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        boxes = load_yolo_boxes(label_path)
        if not boxes:
            continue
        yield img_path, rng.choice(boxes)


# ---------------------------------------------------------------------------
# Cutout extraction
# ---------------------------------------------------------------------------

def extract_cutout(image: Image.Image, mask: np.ndarray, padding: int = 4):
    """Tight-crop around the mask, return (bgr_crop, feathered_alpha_crop)."""
    img_w, img_h = image.size
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1 = max(0, int(xs.min()) - padding)
    y1 = max(0, int(ys.min()) - padding)
    x2 = min(img_w, int(xs.max()) + 1 + padding)
    y2 = min(img_h, int(ys.max()) + 1 + padding)

    crop_rgb = np.array(image.crop((x1, y1, x2, y2)))
    crop_mask = mask[y1:y2, x1:x2].astype(np.float32)
    crop_mask = cv2.GaussianBlur(crop_mask, (7, 7), 0)  # feather edges, avoid hard cutout line
    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    return crop_bgr, crop_mask


def resize_cutout(bgr: np.ndarray, alpha: np.ndarray, target_w: int):
    h, w = bgr.shape[:2]
    scale = target_w / w
    target_h = max(1, int(h * scale))
    bgr_r = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    alpha_r = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return bgr_r, alpha_r


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def make_background(canvas_size: int, background_path: str | None) -> np.ndarray:
    if background_path:
        bg = cv2.imread(background_path)
        if bg is None:
            raise FileNotFoundError(f"Could not read background image: {background_path}")
        return cv2.resize(bg, (canvas_size, canvas_size))
    # neutral checkout-counter-ish flat color as a placeholder background
    canvas = np.full((canvas_size, canvas_size, 3), (190, 195, 198), dtype=np.uint8)
    return canvas


def grid_positions(n: int, canvas_size: int, cell_margin: int, rng: random.Random):
    """Rough grid cells with jitter so objects spread out but can still overlap."""
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell_w = canvas_size // cols
    cell_h = canvas_size // rows
    positions = []
    for i in range(n):
        r, c = divmod(i, cols)
        base_x = c * cell_w + cell_w // 2
        base_y = r * cell_h + cell_h // 2
        jitter_x = rng.randint(-cell_margin, cell_margin)
        jitter_y = rng.randint(-cell_margin, cell_margin)
        positions.append((base_x + jitter_x, base_y + jitter_y))
    return positions


def paste_object(canvas_bgr: np.ndarray, obj_bgr: np.ndarray, obj_alpha: np.ndarray,
                  center_x: int, center_y: int):
    """Alpha-blend obj onto canvas centered at (center_x, center_y). Returns full-canvas
    boolean footprint mask and the clipped (x1, y1, x2, y2) paste region."""
    h, w = obj_alpha.shape
    H, W = canvas_bgr.shape[:2]
    x1, y1 = center_x - w // 2, center_y - h // 2
    x2, y2 = x1 + w, y1 + h

    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(W, x2), min(H, y2)
    if cx2 <= cx1 or cy2 <= cy1:
        return np.zeros((H, W), dtype=bool), (0, 0, 0, 0)

    ox1, oy1 = cx1 - x1, cy1 - y1
    ox2, oy2 = ox1 + (cx2 - cx1), oy1 + (cy2 - cy1)

    region = canvas_bgr[cy1:cy2, cx1:cx2].astype(np.float32)
    o_bgr = obj_bgr[oy1:oy2, ox1:ox2].astype(np.float32)
    a = obj_alpha[oy1:oy2, ox1:ox2][..., None]
    canvas_bgr[cy1:cy2, cx1:cx2] = (region * (1 - a) + o_bgr * a).astype(np.uint8)

    footprint = np.zeros((H, W), dtype=bool)
    footprint[cy1:cy2, cx1:cx2] = obj_alpha[oy1:oy2, ox1:ox2] > 0.5
    return footprint, (cx1, cy1, cx2, cy2)


def resolve_visible_boxes(footprints: list[np.ndarray], visibility_thresh: float):
    """Later-pasted objects occlude earlier ones. Returns list of (bbox or None)
    per object, keeping only ones with enough visible area left."""
    occluded_so_far = np.zeros_like(footprints[0])
    visible_boxes = [None] * len(footprints)
    for i in reversed(range(len(footprints))):
        full = footprints[i]
        vis = full & ~occluded_so_far
        occluded_so_far = occluded_so_far | full
        total = int(full.sum())
        if total == 0:
            continue
        if vis.sum() / total < visibility_thresh:
            continue
        ys, xs = np.where(vis)
        visible_boxes[i] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return visible_boxes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAM3 copy-paste single-scene proof of concept")
    parser.add_argument("--dataset", default="yolo_dataset")
    parser.add_argument("--checkpoint", default="sam3.pt")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-products", type=int, default=7)
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--background", default=None, help="Optional background image path")
    parser.add_argument("--output", default="copy_paste_demo")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--visibility-thresh", type=float, default=0.3,
                         help="Min visible fraction of a pasted object to keep its bbox")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        # sam3 hardcodes device="cuda" in several tensor-creation calls during model
        # construction (position encoding, decoder coords, ...) regardless of the
        # requested device; redirect those calls to cpu wherever they show up
        def _cpu_safe(fn):
            def wrapper(*a, **kw):
                if kw.get("device") == "cuda":
                    kw["device"] = "cpu"
                return fn(*a, **kw)
            return wrapper
        for _name in ("zeros", "ones", "empty", "full", "arange", "tensor", "rand", "randn", "eye", "linspace"):
            setattr(torch, _name, _cpu_safe(getattr(torch, _name)))

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"Loading SAM3 from checkpoint: {args.checkpoint}")
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, device=args.device)
    processor = Sam3Processor(model, device=args.device, confidence_threshold=args.confidence)
    print(f"Model ready. Device: {args.device}\n")

    dataset_root = Path(args.dataset).expanduser()
    cutouts = []  # (source_name, bgr, alpha)
    tried = 0

    pbar = tqdm(total=args.n_products, desc="Extracting cutouts", unit="cutout")
    for img_path, (cx, cy, bw, bh) in iter_candidate_instances(dataset_root, args.split, args.seed):
        if len(cutouts) >= args.n_products:
            break
        tried += 1
        pbar.set_postfix(tried=tried)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            tqdm.write(f"  skip {img_path.name}: cannot open ({e})")
            continue

        W, H = image.size
        box_xyxy = yolo_to_xyxy(cx, cy, bw, bh, W, H)

        try:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                state = processor.set_image(image)
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
            tqdm.write(f"  skip {img_path.name}: no matching mask")
            continue

        result = extract_cutout(image, mask)
        if result is None:
            tqdm.write(f"  skip {img_path.name}: empty mask")
            continue

        bgr, alpha = result
        cutouts.append((img_path.stem, bgr, alpha))
        tqdm.write(f"  [{len(cutouts)}/{args.n_products}] cutout from {img_path.name}")
        pbar.update(1)

    pbar.close()
    if len(cutouts) < 2:
        print(f"Not enough valid cutouts extracted ({len(cutouts)}/{args.n_products} after {tried} images tried), aborting.")
        return

    # resize each cutout to a random size relative to canvas so the scene looks
    # like a real checkout counter rather than one giant + tiny objects
    canvas = make_background(args.canvas_size, args.background)
    positions = grid_positions(len(cutouts), args.canvas_size,
                                cell_margin=args.canvas_size // 12, rng=rng)

    footprints = []
    boxes_info = []  # (source_name, footprint_area)
    min_w = int(args.canvas_size * 0.14)
    max_w = int(args.canvas_size * 0.30)
    for (name, bgr, alpha), (cx, cy) in zip(cutouts, positions):
        target_w = rng.randint(min_w, max_w)
        bgr_r, alpha_r = resize_cutout(bgr, alpha, target_w)
        footprint, _ = paste_object(canvas, bgr_r, alpha_r, cx, cy)
        footprints.append(footprint)
        boxes_info.append(name)

    visible_boxes = resolve_visible_boxes(footprints, args.visibility_thresh)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "composite_clean.jpg"), canvas)

    boxed = canvas.copy()
    kept, dropped = 0, 0
    for name, box in zip(boxes_info, visible_boxes):
        if box is None:
            dropped += 1
            continue
        kept += 1
        x1, y1, x2, y2 = box
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (50, 205, 50), 2)
        cv2.putText(boxed, name[:18], (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 205, 50), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_dir / "composite_boxes.jpg"), boxed)

    print(f"\nDone. {kept} kept / {dropped} dropped (occluded below {args.visibility_thresh:.0%}).")
    print(f"Saved: {output_dir / 'composite_clean.jpg'}")
    print(f"Saved: {output_dir / 'composite_boxes.jpg'}")


if __name__ == "__main__":
    main()
