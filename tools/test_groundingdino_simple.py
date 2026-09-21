#!/usr/bin/env python3
"""
Grounding DINO product detection - Simple test on RPC data.

Usage:
  python test_groundingdino_simple.py

Requirements:
  pip install torch torchvision
  pip install git+https://github.com/IDEA-Research/GroundingDINO.git
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def find_test_images(start_path: str = "."):
    """Find test images directory."""
    # Try common paths
    common_paths = [
        Path(start_path) / "test2019",
        Path(start_path) / "val2019_unlabeled",
        Path(start_path) / "data" / "test",
        Path(start_path) / "dataset" / "test",
    ]
    
    for p in common_paths:
        if p.exists():
            images = list(p.glob("*.jpg")) + list(p.glob("*.png"))
            if images:
                print(f"✅ Found test dir: {p}")
                return p, images
    
    # Fallback: search recursively
    for p in Path(start_path).rglob("*.jpg"):
        if "test" in str(p).lower() or "val" in str(p).lower():
            print(f"✅ Found test image: {p.parent}")
            parent = p.parent
            images = list(parent.glob("*.jpg")) + list(parent.glob("*.png"))
            if images:
                return parent, images
    
    raise FileNotFoundError("Could not find test images directory")


def load_groundingdino_model():
    """Load Grounding DINO."""
    print("🔄 Loading Grounding DINO model...")
    
    try:
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict
    except ImportError:
        print("❌ Missing: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        raise
    
    # Load model
    model_config = "groundingdino/config/GroundingDINO_SwinB.py"
    model_checkpoint = "weights/groundingdino_swinb_cogvlm.pth"
    
    if not Path(model_config).exists():
        raise FileNotFoundError(f"Config not found: {model_config}")
    
    model = build_model(model_config)
    
    if Path(model_checkpoint).exists():
        print(f"Loading weights: {model_checkpoint}")
        checkpoint = torch.load(model_checkpoint, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print("✅ Weights loaded\n")
    else:
        print(f"⚠️  Weights not found: {model_checkpoint}")
        print("   Download from: https://huggingface.co/ShilongLiu/GroundingDINO\n")
    
    model = model.cuda()
    model.eval()
    return model


def run_inference(model, image_path, prompt: str = "product", conf_threshold: float = 0.3):
    """Run Grounding DINO inference."""
    from groundingdino.util.inference import predict
    
    image = Image.open(image_path).convert("RGB")
    
    # Run prediction
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=conf_threshold,
        text_threshold=0.25
    )
    
    return image, boxes, logits, phrases


def draw_boxes_on_image(image: Image.Image, boxes: np.ndarray, logits: np.ndarray,
                       output_path: str = None):
    """Draw bounding boxes on image and save."""
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    # Denormalize boxes (0-1 range -> pixel coordinates)
    boxes_pixel = boxes.copy()
    boxes_pixel[:, 0] *= w  # x1
    boxes_pixel[:, 2] *= w  # x2
    boxes_pixel[:, 1] *= h  # y1
    boxes_pixel[:, 3] *= h  # y2
    
    # Draw boxes
    for i, (box, conf) in enumerate(zip(boxes_pixel, logits)):
        x1, y1, x2, y2 = map(int, box)
        # Green box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label
        label = f"product {conf:.3f}"
        cv2.putText(image_np, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved: {output_path}")
    
    return image_np


def test_random_image(model, test_dir: Path):
    """Test on random image."""
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    random_image = random.choice(images)
    
    print(f"Testing: {random_image.name}\n")
    
    try:
        image, boxes, logits, phrases = run_inference(model, str(random_image))
        
        print(f"Image size: {image.size}")
        print(f"Detections: {len(boxes)}")
        if len(boxes) > 0:
            print(f"Confidences: {logits}")
            print(f"Box coordinates (normalized, xywh): \n{boxes}\n")
        
        # Draw and save
        draw_boxes_on_image(image, boxes, logits, "visualizations/gdino_test.jpg")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    # Find test images
    print("🔍 Searching for test images...\n")
    test_dir, images = find_test_images()
    print(f"Found {len(images)} images\n")
    
    # Load model
    model = load_groundingdino_model()
    
    # Test
    test_random_image(model, test_dir)



if __name__ == "__main__":
    main()
