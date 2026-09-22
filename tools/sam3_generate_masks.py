#!/usr/bin/env python3
"""
SAM3 → COCO RLE mask annotation generation for YOLO segmentation training.

Generates COCO-format JSON with RLE-encoded segmentation masks for all products.
For each image:
  1. Encode image with SAM3 visual backbone (cached)
  2. For each YOLO box, run geometric prompt to extract mask
  3. Encode mask as COCO RLE format
  4. Save combined COCO JSON with masks

Usage:
  python tools/sam3_generate_masks.py --dataset yolo_dataset_rpc --splits train train_pseudo

Requirements:
  pip install git+https://github.com/facebookresearch/sam3.git
  pip install pycocotools
"""

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm


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
    """Convert normalized YOLO box to pixel xyxy."""
    return [
        max(0.0, (cx - w / 2) * img_w),
        max(0.0, (cy - h / 2) * img_h),
        min(float(img_w), (cx + w / 2) * img_w),
        min(float(img_h), (cy + h / 2) * img_h),
    ]


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
    ious = box_iou(box_xyxy, sam3_boxes.float())
    best = int(ious.argmax())
    if ious[best] < iou_thresh:
        return None
    m = sam3_masks[best]
    # SAM3 may return [1, H, W] or [H, W]; always use last two dims
    m = m.reshape(m.shape[-2], m.shape[-1])
    if m.dtype == torch.bool:
        return m.cpu().numpy().astype(np.uint8) * 255
    return ((m > 0).cpu().numpy() * 255).astype(np.uint8)


def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert binary mask to COCO RLE format."""
    # mask: [H, W] uint8 (0 or 255)
    mask_uint8 = (mask > 127).astype(np.uint8)
    rle = mask_utils.encode(np.asfortranarray(mask_uint8))
    return rle


def process_split(split: str, dataset_root: Path, processor, output_path: Path,
                  device: str, iou_threshold: float = 0.1) -> dict:
    """
    Process one split (train, val, test) and generate COCO JSON with RLE masks.
    
    Returns dict with coco_data (images, annotations, categories) ready to save.
    """
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    
    if not images_dir.exists():
        print(f"[{split}] WARN: images dir not found → {images_dir}")
        return {"images": [], "annotations": [], "categories": [{"id": 0, "name": "product"}]}
    
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "product"}],
    }
    
    annotation_id = 0
    device_type = "cuda" if "cuda" in device else "cpu"
    _geo_keys = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
    
    print(f"\n[{split}] Processing {len(image_paths)} images...")
    
    for img_idx, img_path in enumerate(tqdm(image_paths, desc=split)):
        label_path = labels_dir / (img_path.stem + ".txt")
        yolo_boxes = load_yolo_boxes(label_path)
        
        if not yolo_boxes:
            continue
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  Error opening {img_path.name}: {e}")
            continue
        
        W, H = image.size
        
        # Add image entry
        img_id = img_idx
        coco_data["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": W,
            "height": H,
        })
        
        # Encode image with SAM3 once
        try:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                state = processor.set_image(image)
        except Exception as e:
            print(f"  [{split}] Image encode failed {img_path.name}: {e}")
            continue
        
        # Process each YOLO box
        for box_idx, (cx, cy, bw, bh) in enumerate(yolo_boxes):
            box_xyxy = yolo_to_xyxy(cx, cy, bw, bh, W, H)
            
            # Skip tiny boxes
            if (box_xyxy[2] - box_xyxy[0]) < 4 or (box_xyxy[3] - box_xyxy[1]) < 4:
                continue
            
            # Clear geometric state; reuse visual encoding
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
                
                if sam3_masks is not None and sam3_masks.numel() > 0 and sam3_boxes is not None:
                    mask = best_mask_for_box(box_xyxy, sam3_masks, sam3_boxes.float(),
                                            iou_thresh=iou_threshold)
            except Exception as e:
                print(f"  [{split}] SAM3 failed {img_path.name} box {box_idx}: {e}")
            
            # Create annotation entry
            if mask is not None:
                rle = mask_to_rle(mask)
                segmentation = {
                    "size": [H, W],
                    "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
                }
            else:
                # Fallback: use bounding box as segmentation (rectangle)
                x1, y1, x2, y2 = [int(v) for v in box_xyxy]
                # Create rectangular mask
                rect_mask = np.zeros((H, W), dtype=np.uint8)
                rect_mask[y1:y2, x1:x2] = 255
                rle = mask_to_rle(rect_mask)
                segmentation = {
                    "size": [H, W],
                    "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
                }
            
            x1, y1, x2, y2 = box_xyxy
            area = (x2 - x1) * (y2 - y1)
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO format: [x, y, width, height]
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation,
            })
            annotation_id += 1
    
    print(f"[{split}] ✓ Processed {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description="SAM3 → COCO RLE mask annotation generation"
    )
    parser.add_argument(
        "--dataset",
        default="yolo_dataset_rpc",
        help="Root of YOLO dataset (default: yolo_dataset_rpc)",
    )
    parser.add_argument(
        "--checkpoint",
        default="sam3.pt",
        help="Path to SAM3 checkpoint (default: sam3.pt)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "train_pseudo"],
        help="Splits to process (default: train train_pseudo)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Output directory for COCO JSONs (default: current dir)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.1,
        help="Minimum IoU for mask-to-box matching (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Normalize device string
    device_str = args.device
    if device_str.isdigit():
        device_str = f"cuda:{device_str}"
    elif device_str == "cpu":
        device_str = "cpu"
    elif not device_str.startswith("cuda:") and device_str != "cpu":
        print(f"WARNING: Unknown device format '{args.device}', using 'cuda:0'")
        device_str = "cuda:0"
    
    # Load SAM3
    print(f"Loading SAM3 from {args.checkpoint}...")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except ImportError:
        print("ERROR: SAM3 not installed. Install with:")
        print("  pip install git+https://github.com/facebookresearch/sam3.git")
        return
    
    try:
        model = build_sam3_image_model(checkpoint_path=args.checkpoint, device=device_str)
        processor = Sam3Processor(model, device=device_str, confidence_threshold=0.3)
        print(f"✓ SAM3 loaded on {device_str}\n")
    except Exception as e:
        print(f"ERROR loading SAM3: {e}")
        print("Ensure checkpoint exists at:", args.checkpoint)
        return
    
    # Process splits
    dataset_root = Path(args.dataset)
    output_dir = Path(args.output_dir)
    
    for split in args.splits:
        coco_data = process_split(split, dataset_root, processor, output_dir,
                                 args.device, iou_threshold=args.iou_threshold)
        
        # Save JSON
        output_file = output_dir / f"instances_{split}_seg.json"
        with open(output_file, "w") as f:
            json.dump(coco_data, f)
        print(f"✓ Saved {output_file}\n")
    
    print("All done!")


if __name__ == "__main__":
    main()
