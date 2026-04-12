from pathlib import Path

from ultralytics import YOLO

from src.config import Config, get_project_root


def train(cfg: Config, dataset_yaml: str = None) -> str:
    """
    Train a YOLO model on the converted RPC dataset.

    Returns the path to the best model weights.
    """
    project_root = get_project_root()
    tc = cfg.training

    if dataset_yaml is None:
        dataset_yaml = str(project_root / cfg.output.root / "dataset.yaml")

    print(f"Model        : {tc.model}")
    print(f"Dataset YAML : {dataset_yaml}")
    print(f"Epochs       : {tc.epochs}")
    print(f"Image size   : {tc.imgsz}")
    print(f"Batch size   : {tc.batch}")
    print(f"Device       : {tc.device}")
    print()

    model = YOLO(tc.model)

    results = model.train(
        data=dataset_yaml,
        epochs=tc.epochs,
        imgsz=tc.imgsz,
        batch=tc.batch,
        device=tc.device,
        workers=tc.workers,
        optimizer=tc.optimizer,
        lr0=tc.lr0,
        lrf=tc.lrf,
        patience=tc.patience,
        project=str(project_root / tc.project),
        name=tc.name,
        pretrained=tc.pretrained,
        resume=tc.resume,
        exist_ok=tc.exist_ok,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")
    return str(best_weights)
