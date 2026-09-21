#!/usr/bin/env python3
"""
Grounding DINO product detection via Hugging Face transformers.

Simple, clean inference using the transformers library.

Usage:
  python test_groundingdino_simple.py [--num-test N] [--conf THRESHOLD]

Requirements:
  pip install torch torchvision transformers pillow
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def find_test_images(start_path: str = "."):
    """Find test images directory."""
    common_paths = [
        Path(start_path) / "test2019",
        Path(start_path) / "val2019_unlabeled",
        Path(start_path) / "data" / "test",
    ]
    
    for p in common_paths:
        if p.exists():
            images = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
            if images:
                print(f"✅ Found test dir: {p} ({len(images)} images)\n")
                return p, images
    
    raise FileNotFoundError("Could not find test images directory")


def load_model():
    """Load Grounding DINO via transformers."""
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🔄 Loading model from {model_id}...")
    print(f"   Device: {device}\n")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    model.eval()
    
    print("✅ Model loaded\n")
    return model, processor, device


def run_detection(model, processor, device, image_path, prompt: str = "product",
                  box_threshold: float = 0.3):
    """Run detection on a single image."""
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process
    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=0.25,
        target_sizes=target_sizes
    )[0]
    
    return image, results


def draw_and_save(image: Image.Image, results: dict, output_path: str = None):
    """Draw boxes on image and save."""
    image_np = np.array(image)
    
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    boxes = results["boxes"].cpu().numpy()
    
    print(f"Detections: {len(boxes)}")
    if len(boxes) > 0:
        print(f"Confidences: {scores}")
        print(f"Boxes (xyxy in pixels):\n{boxes}\n")
    
    # Draw boxes
    for score, label, box in zip(scores, labels, boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Green rectangle
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label with confidence
        text = f"{label} {score:.3f}"
        cv2.putText(image_np, text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved: {output_path}\n")
    
    return image_np


def test_random_image(model, processor, device, test_dir: Path,
                     box_threshold: float = 0.3):
    """Test on random image."""
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    random_image = random.choice(images)
    
    print(f"Testing: {random_image.name}")
    print(f"Prompt: 'product'\n")
    
    image, results = run_detection(model, processor, device, str(random_image),
                                  prompt="product", box_threshold=box_threshold)
    
    print(f"Image size: {image.size}")
    
    draw_and_save(image, results, "visualizations/gdino_test.jpg")


def evaluate_multiple(model, processor, device, test_dir: Path,
                     num_images: int = 100, box_threshold: float = 0.3):
    """Evaluate on multiple images."""
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    images = images[:num_images]
    
    print(f"Evaluating {len(images)} images with threshold {box_threshold}...\n")
    
    total_detections = 0
    images_with_detections = 0
    
    for i, img_path in enumerate(images):
        try:
            image, results = run_detection(model, processor, device, str(img_path),
                                          prompt="product", box_threshold=box_threshold)
            num_boxes = len(results["boxes"])
            if num_boxes > 0:
                total_detections += num_boxes
                images_with_detections += 1
                
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(images)}")
        except Exception as e:
            print(f"Error on {img_path.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Grounding DINO Evaluation Results")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total boxes: {total_detections}")
    print(f"Avg boxes per image: {total_detections / len(images):.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-test", type=int, default=1,
                       help="Number of images to test (1 = random, >1 = evaluate)")
    parser.add_argument("--conf", type=float, default=0.3,
                       help="Box confidence threshold")
    args = parser.parse_args()
    
    # Find test images
    print("🔍 Searching for test images...\n")
    test_dir, _ = find_test_images()
    
    # Load model
    model, processor, device = load_model()
    
    # Run test
    if args.num_test == 1:
        test_random_image(model, processor, device, test_dir, box_threshold=args.conf)
    else:
        evaluate_multiple(model, processor, device, test_dir,
                         num_images=args.num_test, box_threshold=args.conf)



if __name__ == "__main__":
    main()
