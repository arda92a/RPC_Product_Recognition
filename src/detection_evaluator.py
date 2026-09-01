"""
Pure product-detection evaluation on the RPC test set — isolates the detector's
recall/precision/instance-counting performance from the downstream metric-
learning classifier. Useful because cAcc can never match for an image if the
detector itself misses or double-counts instances, regardless of how good the
classifier is.

Reports, per clutter level (easy/medium/hard) and averaged, using greedy
class-agnostic IoU matching between predicted and ground-truth boxes:
  - recall / precision / F1
  - avg_count_diff: mean(|predicted instance count - GT instance count|) per
    image — a class-agnostic lower bound on the pipeline's ACD.
  - exact_match_rate: fraction of images where predicted count == GT count —
    an optimistic upper bound on what cAcc could ever reach (a count mismatch
    guarantees CD_i > 0, but a matching count can still hide a FN+FP that
    cancel out).
  - perfect_match_rate: fraction of images where every GT instance was
    actually IoU-matched and no spurious boxes were predicted (fn == 0 and
    fp == 0) — the strict, honest per-instance version of exact_match_rate.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.config import Config, get_project_root
from src.converter import build_image_annotation_map, build_image_info_map, load_coco_annotations

LEVEL_ORDER = {"easy": 0, "medium": 1, "hard": 2, "averaged": 3}


def _coco_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return x, y, x + w, y + h


def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """[P, G] pairwise IoU between two sets of xyxy boxes."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    px1, py1, px2, py2 = pred_boxes[:, 0:1], pred_boxes[:, 1:2], pred_boxes[:, 2:3], pred_boxes[:, 3:4]
    gx1, gy1, gx2, gy2 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]

    ix1, iy1 = np.maximum(px1, gx1), np.maximum(py1, gy1)
    ix2, iy2 = np.minimum(px2, gx2), np.minimum(py2, gy2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)

    pred_area = (px2 - px1) * (py2 - py1)
    gt_area = (gx2 - gx1) * (gy2 - gy1)
    union = pred_area + gt_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def _match_boxes(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray,
                  iou_threshold: float) -> Tuple[int, int, int]:
    """Greedy, class-agnostic IoU matching (highest-confidence pred first). Returns (TP, FP, FN)."""
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    order = np.argsort(-pred_scores)
    ious = _iou_matrix(pred_boxes, gt_boxes)
    matched_gt = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    for i in order:
        candidate = np.where(~matched_gt)[0]
        if len(candidate) == 0:
            break
        best_j = candidate[np.argmax(ious[i, candidate])]
        if ious[i, best_j] >= iou_threshold:
            matched_gt[best_j] = True
            tp += 1

    return tp, len(pred_boxes) - tp, len(gt_boxes) - tp


def _score_level(rows: List[dict]) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_count_diff = float(np.mean([abs(r["pred_count"] - r["gt_count"]) for r in rows]))
    exact_match_rate = float(np.mean([r["pred_count"] == r["gt_count"] for r in rows]))
    perfect_match_rate = float(np.mean([r["fp"] == 0 and r["fn"] == 0 for r in rows]))
    return {"recall": recall, "precision": precision, "f1": f1, "avg_count_diff": avg_count_diff,
            "exact_match_rate": exact_match_rate, "perfect_match_rate": perfect_match_rate}


def _print_table(rows: List[dict]):
    order = ["diff", "recall", "precision", "f1", "avg_count_diff", "exact_match_rate", "perfect_match_rate"]
    print("| " + " | ".join(f"{c:>14}" for c in order) + " |")
    print("| " + " | ".join("---:" for _ in order) + " |")
    for row in sorted(rows, key=lambda r: LEVEL_ORDER.get(r["diff"], 9)):
        cells = [row["diff"], f"{row['recall'] * 100:.2f}%", f"{row['precision'] * 100:.2f}%",
                 f"{row['f1'] * 100:.2f}%", f"{row['avg_count_diff']:.2f}", f"{row['exact_match_rate'] * 100:.2f}%",
                 f"{row['perfect_match_rate'] * 100:.2f}%"]
        print("| " + " | ".join(f"{c:>14}" for c in cells) + " |")


def evaluate_detector_counting(cfg: Config, detector_path: str, conf: Optional[float] = None,
                                iou: Optional[float] = None, imgsz: Optional[int] = None,
                                iou_match_threshold: float = 0.5,
                                max_images: Optional[int] = None) -> Dict[str, dict]:
    project_root = get_project_root()
    ec = cfg.evaluation

    conf = ec.conf if conf is None else conf
    iou = ec.iou if iou is None else iou
    imgsz = ec.imgsz if imgsz is None else imgsz

    dataset_root = Path(cfg.dataset.root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    ann_path = dataset_root / cfg.dataset.annotations["test"]
    img_dir = dataset_root / cfg.dataset.images["test"]

    print(f"Detector : {detector_path}")
    print(f"Test ann : {ann_path}")
    print(f"conf={conf} iou={iou} imgsz={imgsz} iou_match_threshold={iou_match_threshold}\n")

    coco = load_coco_annotations(ann_path)
    image_info = build_image_info_map(coco)
    ann_map = build_image_annotation_map(coco)

    file_names = [img["file_name"] for img in image_info.values()]
    levels = {img["file_name"]: img.get("level") for img in image_info.values()}
    has_levels = any(v is not None for v in levels.values())
    if max_images is not None:
        file_names = file_names[:max_images]
    file_name_to_id = {img["file_name"]: img["id"] for img in image_info.values()}

    print("Loading detector ...")
    detector = YOLO(detector_path)

    rows = []
    # one image at a time — see evaluate_pipeline.py: batching a huge path list into
    # ultralytics predict() risks pre-buffering everything and getting OOM-killed.
    for fn in tqdm(file_names, desc="detection-only", unit="img"):
        img_path = str(img_dir / fn)
        result = detector.predict(source=img_path, conf=conf, iou=iou, imgsz=imgsz,
                                   device=ec.device, verbose=False)[0]
        pred_boxes = result.boxes.xyxy.cpu().numpy() if len(result.boxes) else np.empty((0, 4))
        pred_scores = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty((0,))

        img_id = file_name_to_id[fn]
        gt_boxes = np.array([_coco_to_xyxy(a["bbox"]) for a in ann_map.get(img_id, [])],
                             dtype=np.float32).reshape(-1, 4)

        tp, fp, fn_count = _match_boxes(pred_boxes, pred_scores, gt_boxes, iou_match_threshold)
        rows.append({"file_name": fn, "level": levels.get(fn), "tp": tp, "fp": fp, "fn": fn_count,
                     "pred_count": len(pred_boxes), "gt_count": len(gt_boxes)})

    print("\nScoring ...")
    table_rows = [{"diff": "averaged", **_score_level(rows)}]
    if has_levels:
        for level in ("easy", "medium", "hard"):
            level_rows = [r for r in rows if r["level"] == level]
            if level_rows:
                table_rows.append({"diff": level, **_score_level(level_rows)})
    else:
        print("WARNING: ground truth has no per-image 'level' field — only the averaged row is reported.")

    print()
    _print_table(table_rows)

    if cfg.wandb.enabled:
        try:
            import wandb

            wc = cfg.wandb
            run = wandb.init(project=wc.project, name=f"{wc.run_name}-detection-eval",
                              config={"conf": conf, "iou": iou, "imgsz": imgsz,
                                      "iou_match_threshold": iou_match_threshold})
            log_dict = {}
            for row in table_rows:
                for k in ("recall", "precision", "f1", "avg_count_diff", "exact_match_rate", "perfect_match_rate"):
                    log_dict[f"detection/{row['diff']}/{k}"] = row[k]
            run.log(log_dict)
            wandb.finish()
        except ImportError:
            print("WARNING: wandb not installed — skipping W&B logging. Run: pip install wandb")

    return {row["diff"]: row for row in table_rows}
