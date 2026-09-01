"""
Full ACO (Automatic Checkout) pipeline evaluation on the RPC test set:

  1) The YOLO product detector proposes class-agnostic product boxes on each
     checkout image.
  2) Each box is cropped (with the same padding used at training time),
     embedded with the trained metric model, and classified via 1-NN against
     a gallery built from metric_dataset/train.
  3) Predicted per-image category counts are compared against the RPC ground
     truth using the official ACO metrics from the RPC paper (arXiv:1901.07249)
     / rpctool (https://github.com/DIYer22/retail_product_checkout_tools):
     cAcc, ACD, mCCD, mCIoU — reported per clutter level (easy/medium/hard)
     and an "averaged" row over the whole test set.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

from src.augmentor import crop_with_padding
from src.config import Config, get_project_root
from src.converter import load_coco_annotations
from src.metric.backbone import DinoV3Backbone
from src.metric.dataset import IMAGENET_MEAN, IMAGENET_STD, MetricCropDataset
from src.metric.evaluator import extract_embeddings
from src.metric.model import MetricModel

LEVEL_ORDER = {"easy": 0, "medium": 1, "hard": 2, "averaged": 3}


def _sanitize_class_name(name: str) -> str:
    """Must mirror src/augmentor.py's build_class_dirs() folder-name sanitization."""
    return name.replace("/", "_").replace(" ", "_")


def _crop_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _load_metric_model(mc, project_root: Path, checkpoint_path: str, device) -> MetricModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    backbone_checkpoint = None
    if mc.backbone_checkpoint:
        path = Path(mc.backbone_checkpoint)
        backbone_checkpoint = str(path if path.is_absolute() else project_root / path)
    backbone = DinoV3Backbone(mc.backbone, source=mc.backbone_source, hf_model_id=mc.hf_model_id,
                               checkpoint_path=backbone_checkpoint)
    model = MetricModel(backbone, embed_dim=mc.embed_dim, hidden_dim=mc.hidden_dim,
                         dropout=mc.dropout).to(device)
    model.backbone.load_state_dict(checkpoint["backbone"])
    model.head.load_state_dict(checkpoint["head"])
    model.eval()
    return model


def build_gallery(mc, dataset_root: Path, model: MetricModel, device):
    """Embed metric_dataset/train once to serve as the 1-NN reference gallery."""
    train_ds = MetricCropDataset(dataset_root, "train", img_size=mc.img_size)
    loader = DataLoader(train_ds, batch_size=mc.batch_size, shuffle=False, num_workers=mc.num_workers)
    gallery_emb, gallery_labels = extract_embeddings(model, loader, device, desc="gallery")
    return gallery_emb.to(device), gallery_labels.to(device), train_ds.classes


def _build_ground_truth(ann_path: Path, class_to_idx: Dict[str, int]):
    coco = load_coco_annotations(ann_path)
    cat_id_to_idx = {}
    for c in coco["categories"]:
        idx = class_to_idx.get(_sanitize_class_name(c["name"]))
        if idx is not None:
            cat_id_to_idx[c["id"]] = idx
    images = {img["id"]: img for img in coco["images"]}

    gt_counts: Dict[str, List[int]] = defaultdict(list)
    for ann in coco["annotations"]:
        img = images.get(ann["image_id"])
        if img is None:
            continue
        cls_idx = cat_id_to_idx.get(ann["category_id"])
        if cls_idx is None:
            continue  # category not found in metric_dataset/train — shouldn't happen
        gt_counts[img["file_name"]].append(cls_idx)

    levels = {img["file_name"]: img.get("level") for img in images.values()}
    has_levels = any(v is not None for v in levels.values())
    return images, gt_counts, levels, has_levels


@torch.no_grad()
def _classify_boxes(image_bgr, boxes_xyxy, model, transform, gallery_emb, gallery_labels, device,
                     padding_ratio: float, batch_size: int = 64) -> List[int]:
    """Crop each detected box, embed it, and 1-NN classify against the gallery."""
    if len(boxes_xyxy) == 0:
        return []

    crops = []
    for x1, y1, x2, y2 in boxes_xyxy:
        crop = crop_with_padding(image_bgr, [x1, y1, x2 - x1, y2 - y1], padding_ratio)
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops.append(transform(Image.fromarray(crop_rgb)))

    if not crops:
        return []

    preds = []
    for start in range(0, len(crops), batch_size):
        batch = torch.stack(crops[start:start + batch_size]).to(device)
        embeds = model(batch)
        nn_idx = (embeds @ gallery_emb.T).argmax(dim=1)
        preds.extend(gallery_labels[nn_idx].cpu().tolist())
    return preds


def _score_all(pred_counts: Dict[str, List[int]], gt_counts: Dict[str, List[int]],
                K: int, file_names: List[str]) -> Dict[str, float]:
    """Re-implementation of rpctool's calculate(): cAcc, ACD, mCCD, mCIoU."""
    N = len(file_names)
    pred = np.zeros((N, K), dtype=np.float32)
    gt = np.zeros((N, K), dtype=np.float32)
    for i, fn in enumerate(file_names):
        for c in pred_counts.get(fn, []):
            pred[i, c] += 1
        for c in gt_counts.get(fn, []):
            gt[i, c] += 1

    cAcc = float(np.sum(np.all(gt == pred, axis=1)) / N)
    ACD = float(np.sum(np.abs(gt - pred)) / N)

    class_cd = np.sum(np.abs(gt - pred), axis=0)
    class_gt = np.sum(gt, axis=0)
    # categories absent from this subset (class_gt==0) contribute 0 instead of NaN
    mCCD = float(np.sum(np.divide(class_cd, class_gt, out=np.zeros_like(class_cd), where=class_gt > 0)) / K)

    class_min = np.sum(np.minimum(gt, pred), axis=0)
    class_max = np.sum(np.maximum(gt, pred), axis=0)
    mCIoU = float(np.sum(np.divide(class_min, class_max, out=np.zeros_like(class_min), where=class_max > 0)) / K)

    return {"cAcc": cAcc, "ACD": ACD, "mCCD": mCCD, "mCIoU": mCIoU}


def _print_table(rows: List[dict]):
    order = ["diff", "cAcc", "mCIoU", "ACD", "mCCD"]
    print("| " + " | ".join(f"{c:>8}" for c in order) + " |")
    print("| " + " | ".join("---:" for _ in order) + " |")
    for row in sorted(rows, key=lambda r: LEVEL_ORDER.get(r["diff"], 9)):
        cells = [row["diff"], f"{row['cAcc'] * 100:.2f}%", f"{row['mCIoU'] * 100:.2f}%",
                 f"{row['ACD']:.2f}", f"{row['mCCD']:.2f}"]
        print("| " + " | ".join(f"{c:>8}" for c in cells) + " |")


def evaluate_pipeline(cfg: Config, detector_path: str, metric_checkpoint: str,
                       conf: Optional[float] = None, iou: Optional[float] = None,
                       imgsz: Optional[int] = None, max_images: Optional[int] = None) -> Dict[str, dict]:
    project_root = get_project_root()
    ec = cfg.evaluation
    mc = cfg.metric_training

    conf = ec.conf if conf is None else conf
    iou = ec.iou if iou is None else iou
    imgsz = ec.imgsz if imgsz is None else imgsz

    use_cuda = mc.device != "cpu" and torch.cuda.is_available()
    device = torch.device(f"cuda:{mc.device}" if use_cuda else "cpu")

    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    metric_dataset_root = Path(mc.dataset_root)
    if not metric_dataset_root.is_absolute():
        metric_dataset_root = project_root / metric_dataset_root

    ann_path = dataset_root / cfg.dataset.annotations["test"]
    img_dir = dataset_root / cfg.dataset.images["test"]

    print(f"Detector    : {detector_path}")
    print(f"Metric ckpt : {metric_checkpoint}")
    print(f"Test images : {img_dir}")
    print(f"Test ann    : {ann_path}")
    print(f"conf={conf} iou={iou} imgsz={imgsz} device={device}\n")

    print("Loading metric model + building gallery from metric_dataset/train ...")
    metric_model = _load_metric_model(mc, project_root, metric_checkpoint, device)
    gallery_emb, gallery_labels, class_names = build_gallery(mc, metric_dataset_root, metric_model, device)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    K = len(class_names)
    transform = _crop_transform(mc.img_size)
    print(f"Gallery: {gallery_emb.size(0)} crops, {K} categories\n")

    print("Loading ground truth annotations ...")
    images, gt_counts, levels, has_levels = _build_ground_truth(ann_path, class_to_idx)
    file_names = [img["file_name"] for img in images.values()]
    if max_images is not None:
        file_names = file_names[:max_images]
    image_paths = [str(img_dir / fn) for fn in file_names]

    print("Loading detector ...")
    detector = YOLO(detector_path)

    pred_counts: Dict[str, List[int]] = {}
    padding_ratio = cfg.metric_augmentation.padding_ratio

    print(f"Running detection + recognition on {len(image_paths)} test images ...\n")
    results_iter = detector.predict(source=image_paths, conf=conf, iou=iou, imgsz=imgsz,
                                     device=ec.device, stream=True, verbose=False)
    for fn, result in zip(tqdm(file_names, desc="pipeline", unit="img"), results_iter):
        boxes_xyxy = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4))
        pred_counts[fn] = _classify_boxes(result.orig_img, boxes_xyxy, metric_model, transform,
                                           gallery_emb, gallery_labels, device, padding_ratio)

    print("\nScoring ...")
    rows = [{"diff": "averaged", **_score_all(pred_counts, gt_counts, K, file_names)}]

    if has_levels:
        for level in ("easy", "medium", "hard"):
            level_files = [fn for fn in file_names if levels.get(fn) == level]
            if level_files:
                rows.append({"diff": level, **_score_all(pred_counts, gt_counts, K, level_files)})
    else:
        print("WARNING: ground truth has no per-image 'level' field — only the averaged row is reported.")

    print()
    _print_table(rows)

    if cfg.wandb.enabled:
        try:
            import wandb

            wc = cfg.wandb
            run = wandb.init(project=wc.project, name=f"{wc.run_name}-pipeline-eval",
                              config={"conf": conf, "iou": iou, "imgsz": imgsz})
            log_dict = {}
            for row in rows:
                for k in ("cAcc", "mCIoU", "ACD", "mCCD"):
                    log_dict[f"pipeline/{row['diff']}/{k}"] = row[k]
            run.log(log_dict)
            wandb.finish()
        except ImportError:
            print("WARNING: wandb not installed — skipping W&B logging. Run: pip install wandb")

    return {row["diff"]: row for row in rows}
