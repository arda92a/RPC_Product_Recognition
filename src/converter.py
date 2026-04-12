import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import Config, get_project_root


def load_coco_annotations(annotation_path: Path) -> dict:
    """Load a COCO-format JSON annotation file."""
    with open(annotation_path, "r") as f:
        return json.load(f)


def build_image_annotation_map(coco_data: dict) -> Dict[int, List[dict]]:
    """Group annotations by image_id."""
    mapping: Dict[int, List[dict]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        mapping.setdefault(img_id, []).append(ann)
    return mapping


def build_image_info_map(coco_data: dict) -> Dict[int, dict]:
    """Map image_id to image metadata (file_name, width, height)."""
    return {img["id"]: img for img in coco_data["images"]}


def coco_bbox_to_yolo(bbox: List[float], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x_min, y_min, width, height] to YOLO format
    [x_center, y_center, width, height] normalized by image dimensions.
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Clamp to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return x_center, y_center, w_norm, h_norm


def create_yolo_directory_structure(output_root: Path) -> Dict[str, Dict[str, Path]]:
    """
    Create the YOLO directory layout:
        output_root/
            images/train/  images/val/  images/test/
            labels/train/  labels/val/  labels/test/
    """
    paths = {}
    for split in ("train", "val", "test"):
        img_dir = output_root / "images" / split
        lbl_dir = output_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        paths[split] = {"images": img_dir, "labels": lbl_dir}
    return paths


def generate_dataset_yaml(output_root: Path, classes: List[str]) -> Path:
    """Generate the dataset.yaml required by YOLO training."""
    yaml_path = output_root / "dataset.yaml"
    abs_root = output_root.resolve()

    lines = [
        f"path: {abs_root}",
        f"train: images/train",
        f"val: images/val",
        f"test: images/test",
        "",
        f"nc: {len(classes)}",
        f"names: {classes}",
    ]

    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def convert_split(
    split: str,
    annotation_path: Path,
    image_dir: Path,
    output_paths: Dict[str, Path],
    use_symlinks: bool = True,
) -> int:
    """
    Convert one data split (train/val/test) from COCO to YOLO format.

    Returns the number of images processed.
    """
    print(f"[{split}] Loading annotations from {annotation_path}")
    coco_data = load_coco_annotations(annotation_path)

    image_info = build_image_info_map(coco_data)
    ann_map = build_image_annotation_map(coco_data)

    out_img_dir = output_paths["images"]
    out_lbl_dir = output_paths["labels"]
    processed = 0

    for img_id, img_meta in image_info.items():
        file_name = img_meta["file_name"]
        img_w = img_meta["width"]
        img_h = img_meta["height"]

        src_img = image_dir / file_name
        if not src_img.exists():
            print(f"  [WARN] Image not found, skipping: {src_img}")
            continue

        # Link or copy image
        dst_img = out_img_dir / file_name
        if not dst_img.exists():
            if use_symlinks:
                dst_img.symlink_to(src_img.resolve())
            else:
                shutil.copy2(src_img, dst_img)

        # Build YOLO label
        annotations = ann_map.get(img_id, [])
        label_name = Path(file_name).stem + ".txt"
        label_path = out_lbl_dir / label_name

        label_lines = []
        for ann in annotations:
            xc, yc, wn, hn = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            # Single class: always class 0
            label_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        label_path.write_text("\n".join(label_lines) + "\n" if label_lines else "")
        processed += 1

    print(f"[{split}] Processed {processed}/{len(image_info)} images")
    return processed


def convert_dataset(cfg: Config, use_symlinks: bool = True) -> Path:
    """
    Full pipeline: convert the RPC dataset from COCO format to YOLO format.

    Returns the path to the generated dataset.yaml.
    """
    project_root = get_project_root()
    dataset_root = project_root / cfg.dataset.root
    output_root = project_root / cfg.output.root

    print(f"Dataset root : {dataset_root}")
    print(f"Output root  : {output_root}")
    print(f"Classes      : {cfg.classes}")
    print(f"Symlinks     : {use_symlinks}")
    print()

    dir_paths = create_yolo_directory_structure(output_root)

    for split in ("train", "val", "test"):
        ann_path = dataset_root / cfg.dataset.annotations[split]
        img_dir = dataset_root / cfg.dataset.images[split]

        if not ann_path.exists():
            print(f"[{split}] Annotation file not found: {ann_path}, skipping.")
            continue
        if not img_dir.exists():
            print(f"[{split}] Image directory not found: {img_dir}, skipping.")
            continue

        convert_split(split, ann_path, img_dir, dir_paths[split], use_symlinks)

    yaml_path = generate_dataset_yaml(output_root, cfg.classes)
    print(f"\nDataset YAML written to: {yaml_path}")
    return yaml_path
