#!/usr/bin/env python3
"""
SAM3 product segmentation visualization for RPC YOLO dataset.

Uses the native facebookresearch/sam3 package with a local .pt checkpoint.
For each split (train / val / test):
  - Encodes each image once with SAM3
  - Uses text prompt "product" to detect all product instances
  - Matches SAM3 masks to YOLO gt-boxes by box IoU
  - Saves <n_crops> padded product-crop visualizations with mask overlay

Usage on server (from project root  ~/01-code/RetailProject/RPC_Product_Recognition):
  python sam3_segment.py                          # uses defaults below
  python sam3_segment.py --dataset /other/path/yolo_dataset --n-crops 200

Requirements:
  pip install git+https://github.com/facebookresearch/sam3.git
  pip install iopath torchvision
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# YOLO helpers
# ---------------------------------------------------------------------------

def load_yolo_boxes(label_path: Path) -> list[list[float]]:
    """Return list of [cx, cy, w, h] (normalized) from a YOLO label file."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append([float(x) for x in parts[1:5]])
    return boxes


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int, img_h: int) -> list[float]:
    return [
        max(0.0, (cx - w / 2) * img_w),
        max(0.0, (cy - h / 2) * img_h),
        min(float(img_w), (cx + w / 2) * img_w),
        min(float(img_h), (cy + h / 2) * img_h),
    ]


# ---------------------------------------------------------------------------
# Mask matching
# ---------------------------------------------------------------------------

def box_iou(box: list[float], boxes_tensor: torch.Tensor) -> torch.Tensor:
    """IoU between one box [x1,y1,x2,y2] and N boxes tensor [N,4]."""
    b = torch.tensor(box, dtype=torch.float32)
    ix1 = torch.max(b[0], boxes_tensor[:, 0])
    iy1 = torch.max(b[1], boxes_tensor[:, 1])
    ix2 = torch.min(b[2], boxes_tensor[:, 2])
    iy2 = torch.min(b[3], boxes_tensor[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    a1 = (b[2] - b[0]) * (b[3] - b[1])
    a2 = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
    return inter / (a1 + a2 - inter).clamp(1e-6)


def best_mask_for_box(
    box_xyxy: list[float],
    sam3_masks: torch.Tensor,
    sam3_boxes: torch.Tensor,
    iou_thresh: float = 0.1,
) -> np.ndarray | None:
    """Return the SAM3 mask with highest box-IoU to box_xyxy, or None."""
    if sam3_masks is None or sam3_masks.numel() == 0:
        return None
    ious = box_iou(box_xyxy, sam3_boxes)
    best = int(ious.argmax())
    if ious[best] < iou_thresh:
        return None
    m = sam3_masks[best]
    # SAM3 may return [1, H, W] or [H, W]; always use last two dims
    m = m.reshape(m.shape[-2], m.shape[-1])
    if m.dtype == torch.bool:
        return m.cpu().numpy().astype(np.uint8)
    return (m > 0).cpu().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def make_crop_visualization(
    image: Image.Image,
    box_xyxy: list[float],
    mask: np.ndarray | None,
    padding: int = 15,
) -> Image.Image:
    """Crop to product bbox (+padding) and overlay SAM3 mask. Returns RGB image."""
    img_w, img_h = image.size
    x1, y1, x2, y2 = (int(v) for v in box_xyxy)
    cx1, cy1 = max(0, x1 - padding), max(0, y1 - padding)
    cx2, cy2 = min(img_w, x2 + padding), min(img_h, y2 + padding)

    crop = image.crop((cx1, cy1, cx2, cy2)).convert("RGBA")

    if mask is not None:
        mask_crop = mask[cy1:cy2, cx1:cx2]
        overlay = Image.new("RGBA", crop.size, (50, 205, 50, 0))
        overlay.putalpha(Image.fromarray((mask_crop * 140).astype(np.uint8)))
        crop = Image.alpha_composite(crop, overlay)

    draw = ImageDraw.Draw(crop)
    draw.rectangle([x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1],
                   outline=(50, 205, 50, 255), width=2)
    return crop.convert("RGB")


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(split, dataset_root, processor, n_crops, output_dir, device):
    from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Proc
    assert isinstance(processor, _Sam3Proc)

    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    out_dir = output_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"[{split}] WARN: images dir not found → {images_dir}")
        return 0

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    random.shuffle(image_paths)

    saved = 0
    visited = 0
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    # Keys to clear between products; preserves image + language ("visual") features
    _geo_keys = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]

    for img_path in image_paths:
        if saved >= n_crops:
            break

        label_path = labels_dir / (img_path.stem + ".txt")
        yolo_boxes = load_yolo_boxes(label_path)
        if not yolo_boxes:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Could not open {img_path.name}: {e}")
            continue

        W, H = image.size
        visited += 1

        # Encode image once (expensive visual backbone pass)
        try:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                state = processor.set_image(image)
        except Exception as e:
            print(f"  [{split}] Image encode failed {img_path.name}: {e}")
            continue

        # Per-product: YOLO cx,cy,w,h (normalized) → add_geometric_prompt → mask
        for i, (cx, cy, bw, bh) in enumerate(yolo_boxes):
            if saved >= n_crops:
                break

            box_xyxy = yolo_to_xyxy(cx, cy, bw, bh, W, H)
            if (box_xyxy[2] - box_xyxy[0]) < 4 or (box_xyxy[3] - box_xyxy[1]) < 4:
                continue

            # Clear geometric state only; "visual" language features reused after first call
            for key in _geo_keys:
                state.pop(key, None)

            mask = None
            try:
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    output = processor.add_geometric_prompt(
                        box=[cx, cy, bw, bh], label=True, state=state
                    )
                sam3_masks = output.get("masks")
                sam3_boxes = output.get("boxes")
                if sam3_masks is not None and sam3_masks.numel() > 0 \
                        and sam3_boxes is not None:
                    mask = best_mask_for_box(box_xyxy, sam3_masks, sam3_boxes.float())
            except Exception as e:
                print(f"  [{split}] SAM3 failed {img_path.name} box {i}: {e}")

            try:
                vis = make_crop_visualization(image, box_xyxy, mask)
                out_path = out_dir / f"{img_path.stem}_prod{i:03d}.jpg"
                vis.save(out_path, quality=95)
                saved += 1
            except Exception as e:
                print(f"  [{split}] Vis error {img_path.name} box {i}: {e}")

        if visited % 10 == 0:
            print(f"  [{split}] images visited: {visited}, crops saved: {saved}/{n_crops}")

    print(f"[{split}] ✓ Saved {saved} crops from {visited} images → {out_dir}")
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 product segmentation visualization — RPC YOLO dataset"
    )
    parser.add_argument(
        "--dataset",
        default="yolo_dataset",
        help="Root of the single-class YOLO dataset (default: yolo_dataset)",
    )
    parser.add_argument(
        "--checkpoint",
        default="sam3.pt",
        metavar="PATH",
        help="Path to local SAM3 .pt checkpoint (default: sam3.pt)",
    )
    parser.add_argument(
        "--output",
        default="sam3_seg_results",
        help="Output root directory (default: sam3_seg_results)",
    )
    parser.add_argument(
        "--n-crops",
        type=int,
        default=100,
        help="Product crop visualizations to save per split (default: 100)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="SAM3 detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ------------------------------------------------------------------
    # Load SAM3 (native package)
    # ------------------------------------------------------------------
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    ckpt = str(Path(args.checkpoint).expanduser())
    print(f"Loading SAM3 from checkpoint: {ckpt}")
    model = build_sam3_image_model(checkpoint_path=ckpt, device=args.device)
    processor = Sam3Processor(model, device=args.device,
                               confidence_threshold=args.confidence)
    print(f"Model ready. Device: {args.device}\n")

    # ------------------------------------------------------------------
    # Process splits
    # ------------------------------------------------------------------
    dataset_root = Path(args.dataset).expanduser()
    output_dir = Path(args.output)

    for split in args.splits:
        print(f"--- Processing split: {split} ---")
        process_split(split, dataset_root, processor, args.n_crops, output_dir, args.device)

    print(f"\nAll done! Results saved under: {args.output}/")


if __name__ == "__main__":
    main()
