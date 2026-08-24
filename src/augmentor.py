"""
Build a per-class crop dataset for metric learning from the RPC COCO annotations,
augmenting train crops to bridge the studio/checkout domain gap.

Layout produced:
    metric_dataset/train/<category_name>/*.jpg   (1 original + N augmented copies)
    metric_dataset/val/<category_name>/*.jpg     (clean crops only)
    metric_dataset/test/<category_name>/*.jpg    (clean crops only)
"""

import random
from pathlib import Path
from typing import Dict, List

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from src.config import Config, MetricAugmentationConfig, get_project_root
from src.converter import build_image_annotation_map, build_image_info_map, load_coco_annotations


def build_standard_pipeline(mcfg: MetricAugmentationConfig) -> A.Compose:
    """Geometric + photometric + occlusion transforms simulating checkout-camera conditions."""
    return A.Compose([
        A.HorizontalFlip(p=mcfg.horizontal_flip_prob),
        A.Affine(
            scale=(mcfg.scale_jitter_min, mcfg.scale_jitter_max),
            rotate=(-mcfg.rotate_limit, mcfg.rotate_limit),
            p=0.8,
        ),
        A.Perspective(scale=(0.02, 0.08), p=mcfg.perspective_prob),
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=mcfg.brightness_contrast_limit,
                contrast_limit=mcfg.brightness_contrast_limit,
            ),
            A.HueSaturationValue(
                hue_shift_limit=mcfg.hue_shift_limit,
                sat_shift_limit=mcfg.sat_shift_limit,
                val_shift_limit=10,
            ),
        ], p=mcfg.color_jitter_prob),
        A.OneOf([
            A.MotionBlur(blur_limit=mcfg.blur_limit),
            A.GaussianBlur(blur_limit=mcfg.blur_limit),
        ], p=mcfg.blur_prob),
        A.GaussNoise(p=mcfg.noise_prob),
        A.CoarseDropout(
            max_holes=mcfg.cutout_max_holes,
            min_holes=1,
            max_height=mcfg.cutout_max_size_ratio,
            max_width=mcfg.cutout_max_size_ratio,
            fill_value=0,
            p=mcfg.cutout_prob,
        ),
    ])


def crop_with_padding(image: np.ndarray, bbox: List[float], padding_ratio: float) -> np.ndarray:
    """Crop a COCO xywh bbox region out of an image with extra context padding."""
    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox
    pad_w, pad_h = w * padding_ratio, h * padding_ratio
    x0 = max(0, int(x - pad_w))
    y0 = max(0, int(y - pad_h))
    x1 = min(img_w, int(x + w + pad_w))
    y1 = min(img_h, int(y + h + pad_h))
    return image[y0:y1, x0:x1].copy()


def load_background_pool(cfg: Config, mcfg: MetricAugmentationConfig, project_root: Path) -> List[Path]:
    """Raw scene images used as copy-paste backgrounds (real cluttered checkout scenes)."""
    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    bg_dir = dataset_root / cfg.dataset.images[mcfg.copy_paste_background_split]
    if not bg_dir.exists():
        return []
    return sorted(bg_dir.glob("*.jpg"))


def _random_background_patch(bg_paths: List[Path], out_w: int, out_h: int) -> np.ndarray:
    for _ in range(10):
        bg = cv2.imread(str(random.choice(bg_paths)))
        if bg is None:
            continue
        bh, bw = bg.shape[:2]
        if bw < out_w or bh < out_h:
            bg = cv2.resize(bg, (max(out_w, bw), max(out_h, bh)))
            bh, bw = bg.shape[:2]
        x0 = random.randint(0, bw - out_w)
        y0 = random.randint(0, bh - out_h)
        return bg[y0:y0 + out_h, x0:x0 + out_w].copy()
    return np.full((out_h, out_w, 3), 127, dtype=np.uint8)


def _feathered_mask(h: int, w: int, feather: int) -> np.ndarray:
    """Alpha mask that is 1 in the interior and tapers to 0 near the edges."""
    feather = max(3, feather | 1)  # odd kernel size
    mask = np.ones((h, w), dtype=np.float32)
    mask = cv2.copyMakeBorder(mask, feather, feather, feather, feather, cv2.BORDER_CONSTANT, value=0)
    mask = cv2.GaussianBlur(mask, (feather, feather), 0)
    return mask[feather:feather + h, feather:feather + w]


def copy_paste(crop_bgr: np.ndarray, bg_paths: List[Path], mcfg: MetricAugmentationConfig) -> np.ndarray:
    """Paste the product crop onto a real checkout background with feathered edge blending."""
    ch, cw = crop_bgr.shape[:2]
    canvas_w, canvas_h = int(cw * 1.6), int(ch * 1.6)
    background = _random_background_patch(bg_paths, canvas_w, canvas_h)

    scale = random.uniform(0.6, 0.95)
    new_w = max(1, int(canvas_w * scale))
    new_h = max(1, int(canvas_h * scale))
    product = cv2.resize(crop_bgr, (new_w, new_h))
    mask = _feathered_mask(new_h, new_w, mcfg.copy_paste_edge_feather)[..., None]

    x0 = random.randint(0, canvas_w - new_w)
    y0 = random.randint(0, canvas_h - new_h)

    roi = background[y0:y0 + new_h, x0:x0 + new_w].astype(np.float32)
    blended = mask * product.astype(np.float32) + (1 - mask) * roi
    background[y0:y0 + new_h, x0:x0 + new_w] = blended.astype(np.uint8)
    return background


def make_preview_grid(orig: np.ndarray, variants: List[np.ndarray], label_height: int = 22) -> np.ndarray:
    """Tile [orig | aug0 | aug1 | ...] into one labeled strip for visual inspection."""
    tiles = [("orig", orig)] + [(f"aug{i}", v) for i, v in enumerate(variants)]
    h, w = orig.shape[:2]
    panels = []
    for label, tile in tiles:
        panel = np.full((h + label_height, w, 3), 255, dtype=np.uint8)
        panel[label_height:, :, :] = tile
        cv2.putText(panel, label, (4, label_height - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


def build_class_dirs(output_root: Path, split: str, class_names: List[str]) -> Dict[str, Path]:
    dirs = {}
    for name in class_names:
        safe_name = name.replace("/", "_").replace(" ", "_")
        d = output_root / split / safe_name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs


def process_split_for_metric(
    split: str,
    dataset_root: Path,
    output_root: Path,
    cfg: Config,
    mcfg: MetricAugmentationConfig,
    aug_pipeline: A.Compose,
    bg_paths: List[Path],
    preview_counter: List[int],
) -> int:
    ann_path = dataset_root / cfg.dataset.annotations[split]
    img_dir = dataset_root / cfg.dataset.images[split]
    coco_data = load_coco_annotations(ann_path)

    image_info = build_image_info_map(coco_data)
    ann_map = build_image_annotation_map(coco_data)
    cat_names = {c["id"]: c["name"] for c in coco_data["categories"]}
    class_dirs = build_class_dirs(output_root, split, sorted(set(cat_names.values())))

    is_train = split == "train"
    do_augment = is_train and mcfg.enabled
    saved = 0

    preview_dir = None
    if do_augment and mcfg.preview_enabled:
        preview_dir = output_root / mcfg.preview_dir
        preview_dir.mkdir(parents=True, exist_ok=True)

    for img_id, anns in tqdm(ann_map.items(), desc=f"[{split}] extracting crops"):
        img_meta = image_info[img_id]
        img_path = img_dir / img_meta["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        for ann in anns:
            cat_name = cat_names.get(ann["category_id"])
            if cat_name is None:
                continue

            crop = crop_with_padding(image, ann["bbox"], mcfg.padding_ratio)
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (mcfg.img_size, mcfg.img_size))

            out_dir = class_dirs[cat_name]
            stem = f"{img_id}_{ann['id']}"

            if mcfg.keep_original or not is_train:
                cv2.imwrite(str(out_dir / f"{stem}_orig.jpg"), crop)
                saved += 1

            if do_augment:
                want_preview = preview_dir is not None and preview_counter[0] < mcfg.preview_samples
                preview_variants = []
                for k in range(mcfg.copies_per_train_image):
                    variant = crop
                    if bg_paths and random.random() < mcfg.copy_paste_prob:
                        variant = copy_paste(variant, bg_paths, mcfg)
                    variant_rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
                    variant_rgb = aug_pipeline(image=variant_rgb)["image"]
                    variant = cv2.cvtColor(variant_rgb, cv2.COLOR_RGB2BGR)
                    variant = cv2.resize(variant, (mcfg.img_size, mcfg.img_size))
                    cv2.imwrite(str(out_dir / f"{stem}_aug{k}.jpg"), variant)
                    saved += 1
                    if want_preview:
                        preview_variants.append(variant)

                if want_preview and preview_variants:
                    grid = make_preview_grid(crop, preview_variants)
                    safe_cat = cat_name.replace("/", "_").replace(" ", "_")
                    cv2.imwrite(str(preview_dir / f"{safe_cat}_{stem}_grid.jpg"), grid)
                    preview_counter[0] += 1

    return saved


def build_metric_dataset(cfg: Config, splits: List[str] = None) -> Path:
    """Extract + augment per-class crops for metric learning. Returns the output root."""
    mcfg = cfg.metric_augmentation
    project_root = get_project_root()

    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    output_root = Path(mcfg.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root

    random.seed(mcfg.seed)
    np.random.seed(mcfg.seed)

    bg_paths = load_background_pool(cfg, mcfg, project_root)
    aug_pipeline = build_standard_pipeline(mcfg)

    print(f"Dataset root : {dataset_root}")
    print(f"Output root  : {output_root}")
    print(f"Copies/train : {mcfg.copies_per_train_image} (enabled={mcfg.enabled})")
    print(f"Copy-paste   : p={mcfg.copy_paste_prob}, backgrounds={len(bg_paths)} "
          f"(from '{mcfg.copy_paste_background_split}')")
    if not bg_paths:
        print("[WARN] No background images found for copy-paste — that augmentation will be skipped.")
    print()

    splits = splits or ["train", "val", "test"]
    preview_counter = [0]
    total = 0
    for split in splits:
        ann_path = dataset_root / cfg.dataset.annotations[split]
        img_dir = dataset_root / cfg.dataset.images[split]
        if not ann_path.exists() or not img_dir.exists():
            print(f"[{split}] Annotations or images not found, skipping.")
            continue
        n = process_split_for_metric(split, dataset_root, output_root, cfg, mcfg, aug_pipeline, bg_paths, preview_counter)
        print(f"[{split}] Saved {n} crops")
        total += n

    print(f"\nTotal crops saved: {total}")
    if mcfg.preview_enabled and preview_counter[0] > 0:
        print(f"Preview grids ({preview_counter[0]}): {output_root / mcfg.preview_dir}")
    return output_root
