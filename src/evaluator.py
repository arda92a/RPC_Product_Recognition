from pathlib import Path

from ultralytics import YOLO

from src.config import Config, get_project_root


def evaluate(cfg: Config, model_path: str = None) -> dict:
    """
    Evaluate a trained YOLO model on the test (or specified) split.

    Returns the metrics dictionary.
    """
    project_root = get_project_root()
    ec = cfg.evaluation

    if model_path is None:
        model_path = str(project_root / ec.model)

    dataset_yaml = str(project_root / cfg.output.root / "dataset.yaml")

    print(f"Model   : {model_path}")
    print(f"Dataset : {dataset_yaml}")
    print(f"Split   : {ec.split}")
    print(f"Conf    : {ec.conf}")
    print(f"IoU     : {ec.iou}")
    print()

    model = YOLO(model_path)

    metrics = model.val(
        data=dataset_yaml,
        split=ec.split,
        conf=ec.conf,
        iou=ec.iou,
        imgsz=ec.imgsz,
        device=ec.device,
    )

    print("\n--- Evaluation Results ---")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")

    return {
        "map50": metrics.box.map50,
        "map50_95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
    }
