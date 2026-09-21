#!/usr/bin/env python3
"""
Grounding DINO product detection - Simple test on RPC data.

Detects products in images using vision-language model (Grounding DINO).

Usage:
  python test_groundingdino_simple.py  # Random image from test dir
  python test_groundingdino_simple.py --evaluate --max-images 100

Requirements:
  pip install supervision torch torchvision
  pip install git+https://github.com/IDEA-Research/GroundingDINO.git
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


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


def load_groundingdino_model(device: str = "cuda"):
    """Load Grounding DINO."""
    try:
        import supervision as sv
    except ImportError:
        print("❌ Missing: pip install supervision")
        raise

    print("🔄 Loading Grounding DINO model...")
    model = sv.GroundingDINO.from_pretrained(
        "GroundingDINO/groundingdino-b",
        device=device
    )
    print("✅ Model loaded\n")
    return model


def detect_products(model, image_path: str, prompt: str = "product", 
                   confidence_threshold: float = 0.3):
    """Run detection on single image."""
    import supervision as sv
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect
    detections = model.detect(image=image, text=prompt)
    
    # Filter by confidence
    mask = detections.confidence > confidence_threshold
    detections.xyxy = detections.xyxy[mask]
    detections.confidence = detections.confidence[mask]
    
    return image, detections


def visualize_and_save(image, detections, output_path: str):
    """Draw boxes and save."""
    import supervision as sv
    
    # Annotate
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    
    annotated = box_annotator.annotate(scene=image, detections=detections)
    labels = [f"product {conf:.2f}" for conf in detections.confidence]
    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    
    return output_path


def test_single_random_image(model, test_dir: Path, output_dir: str = "visualizations"):
    """Test on random image from test set."""
    images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    random_image = random.choice(images)
    
    print(f"Testing random image: {random_image.name}")
    print(f"Image shape: {cv2.imread(str(random_image)).shape}")
    
    image, detections = detect_products(model, str(random_image), confidence_threshold=0.3)
    
    print(f"✅ Detections found: {len(detections)}")
    if len(detections) > 0:
        print(f"   Confidences: {detections.confidence}")
        print(f"   Boxes (xyxy): {detections.xyxy[:3]}...")  # Show first 3
    
    # Visualize
    output_file = Path(output_dir) / f"gdino_{random_image.stem}.jpg"
    visualize_and_save(image, detections, str(output_file))
    print(f"\n✅ Saved: {output_file}\n")
    
    return len(detections)


def evaluate_test_set(model, test_dir: Path, confidence_threshold: float = 0.3,
                      max_images: int = None):
    """Evaluate on multiple images."""
    images = (list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png")))[:max_images]
    
    print(f"Evaluating {len(images)} images...\n")
    
    total_boxes = 0
    images_with_detections = 0
    
    for image_path in tqdm(images, desc="Processing"):
        try:
            image, detections = detect_products(
                model, str(image_path),
                confidence_threshold=confidence_threshold
            )
            if len(detections) > 0:
                total_boxes += len(detections)
                images_with_detections += 1
        except Exception as e:
            print(f"Error on {image_path.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Grounding DINO Evaluation Results")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total boxes detected: {total_boxes}")
    print(f"Avg boxes per image: {total_boxes / len(images):.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test Grounding DINO on RPC dataset"
    )
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on multiple images")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Find test dir
    print("🔍 Searching for test images...\n")
    test_dir, images = find_test_images()
    print(f"Found {len(images)} images\n")

    # Load model
    model = load_groundingdino_model(device=args.device)

    # Run test
    if args.evaluate:
        evaluate_test_set(model, test_dir, confidence_threshold=args.confidence,
                         max_images=args.max_images)
    else:
        test_single_random_image(model, test_dir, output_dir="visualizations")


if __name__ == "__main__":
    main()
