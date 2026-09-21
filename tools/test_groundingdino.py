#!/usr/bin/env python3
"""
Grounding DINO product detection on RPC test set.

Grounding DINO is a vision-language model that can detect objects
from free-form text prompts (e.g., "product", "item", "box").

Usage (single image visualization):
  python tools/test_groundingdino.py --image-path test2019_images/image123.jpg --visualize

Usage (full test set evaluation):
  python tools/test_groundingdino.py --evaluate-test-set

Requirements:
  pip install groundingdino-py
  pip install transformers supervision
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_groundingdino_model(device: str = "cuda"):
    """Load Grounding DINO model from Hugging Face."""
    try:
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        import groundingdino.datasets.transforms as T
    except ImportError:
        print("Installing Grounding DINO...")
        import subprocess
        subprocess.check_call([
            "pip", "install", "git+https://github.com/IDEA-Research/GroundingDINO.git"
        ])
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        import groundingdino.datasets.transforms as T

    # Config
    config_file = "groundingdino/config/GroundingDINO_SwinB.py"
    checkpoint_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0rc1/groundingdino_swinb_cogvlm.pth"
    
    print(f"Loading Grounding DINO from {checkpoint_url}...")
    
    # For simplicity, use pre-built model from supervision library
    try:
        import supervision as sv
        model = sv.GroundingDINO.from_pretrained("GroundingDINO/groundingdino-b", device=device)
        return model
    except Exception as e:
        print(f"Falling back to manual load: {e}")
        # Manual load (requires more setup)
        raise NotImplementedError("Please install: pip install supervision groundingdino-py")


def load_image(image_path: str):
    """Load image as RGB."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_products(model, image, prompt: str = "product", confidence_threshold: float = 0.3):
    """Run Grounding DINO detection on image."""
    try:
        import supervision as sv
    except ImportError:
        raise ImportError("pip install supervision")

    # Detect with model
    detections = model.detect(image=image, text=prompt)
    
    # Filter by confidence
    mask = detections.confidence > confidence_threshold
    detections.xyxy = detections.xyxy[mask]
    detections.confidence = detections.confidence[mask]
    detections.class_id = detections.class_id[mask]
    
    return detections


def visualize_detections(image, detections, output_path: str = None):
    """Draw bboxes on image and save/display."""
    try:
        import supervision as sv
    except ImportError:
        raise ImportError("pip install supervision")

    # Create box annotator
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Annotate image
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    
    # Add labels (confidence scores)
    labels = [f"{conf:.2f}" for conf in detections.confidence]
    annotated_image = label_annotator.annotate(
        scene=annotated_image, 
        detections=detections, 
        labels=labels
    )

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"✅ Saved visualization: {output_path}")

    return annotated_image


def test_single_image(model, image_path: str, output_dir: str = "visualizations"):
    """Test on single image with visualization."""
    print(f"\n{'='*60}")
    print(f"Testing Grounding DINO on: {image_path}")
    print(f"{'='*60}\n")

    # Load and detect
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")

    detections = detect_products(model, image, prompt="product", confidence_threshold=0.3)
    print(f"Detections found: {len(detections)}")
    print(f"Confidences: {detections.confidence}")

    # Visualize
    output_file = Path(output_dir) / f"groundingdino_{Path(image_path).stem}_viz.jpg"
    visualize_detections(image, detections, output_path=str(output_file))

    return detections


def evaluate_test_set(model, test_images_dir: str, gt_json_path: str, 
                     confidence_threshold: float = 0.3, max_images: int = None):
    """Evaluate Grounding DINO on test set against ground truth."""
    
    test_dir = Path(test_images_dir)
    image_files = sorted(test_dir.glob("*.jpg"))[:max_images]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {test_images_dir}")

    # Load ground truth
    with open(gt_json_path) as f:
        coco_data = json.load(f)
    
    # Build GT dict: {image_name: [boxes in xyxy]}
    gt_by_image = {}
    img_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        filename = Path(img_id_to_filename[img_id]).name
        bbox_coco = ann['bbox']  # [x, y, w, h]
        x1, y1, w, h = bbox_coco
        xyxy = [x1, y1, x1 + w, y1 + h]
        
        if filename not in gt_by_image:
            gt_by_image[filename] = []
        gt_by_image[filename].append(xyxy)

    # Run detection on all images
    tp_total, fp_total, fn_total = 0, 0, 0
    precision_list, recall_list = [], []
    iou_threshold = 0.5

    for image_path in tqdm(image_files, desc="Evaluating test set"):
        try:
            image = load_image(str(image_path))
            detections = detect_products(model, image, prompt="product", 
                                        confidence_threshold=confidence_threshold)
            
            filename = image_path.name
            gt_boxes = np.array(gt_by_image.get(filename, []))
            pred_boxes = detections.xyxy
            
            # Simple IoU matching
            if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                tp = 0
                for gt_box in gt_boxes:
                    # Find best match
                    ious = [compute_iou(gt_box, pb) for pb in pred_boxes]
                    if max(ious) > iou_threshold:
                        tp += 1
                
                fp = len(pred_boxes) - tp
                fn = len(gt_boxes) - tp
            else:
                tp = 0
                fp = len(pred_boxes)
                fn = len(gt_boxes)

            tp_total += tp
            fp_total += fp
            fn_total += fn

            if len(gt_boxes) > 0:
                recall = tp / len(gt_boxes)
                recall_list.append(recall)
            
            if len(pred_boxes) > 0:
                precision = tp / len(pred_boxes)
                precision_list.append(precision)

        except Exception as e:
            print(f"Error on {image_path.name}: {e}")
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"Grounding DINO Test Set Evaluation")
    print(f"{'='*60}")
    print(f"Images evaluated: {len(image_files)}")
    print(f"Total TP: {tp_total}")
    print(f"Total FP: {fp_total}")
    print(f"Total FN: {fn_total}")
    print(f"Overall Precision: {tp_total / (tp_total + fp_total + 1e-5):.4f}")
    print(f"Overall Recall: {tp_total / (tp_total + fn_total + 1e-5):.4f}")
    if precision_list:
        print(f"Mean Precision (per-image): {np.mean(precision_list):.4f}")
    if recall_list:
        print(f"Mean Recall (per-image): {np.mean(recall_list):.4f}")


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Test Grounding DINO on RPC product detection"
    )
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to single test image")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization of detections")
    parser.add_argument("--evaluate-test-set", action="store_true",
                        help="Evaluate on full test set")
    parser.add_argument("--test-images-dir", type=str,
                        default="/home/aoztuner/01-code/RetailProject/RPC_Product_Recognition/test2019",
                        help="Path to test images directory")
    parser.add_argument("--gt-json", type=str,
                        default="/home/aoztuner/01-code/RetailProject/RPC_Product_Recognition/instances_test.json",
                        help="Path to COCO GT JSON")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    # Load model
    print("Loading Grounding DINO model...")
    try:
        model = load_groundingdino_model(device=args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nTry installing:")
        print("  pip install supervision")
        print("  pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'")
        return

    # Single image test
    if args.image_path and args.visualize:
        test_single_image(model, args.image_path, output_dir="visualizations")

    # Full test set evaluation
    if args.evaluate_test_set:
        evaluate_test_set(model, args.test_images_dir, args.gt_json,
                         confidence_threshold=args.confidence,
                         max_images=args.max_images)


if __name__ == "__main__":
    main()
