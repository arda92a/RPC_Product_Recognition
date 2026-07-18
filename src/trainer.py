from pathlib import Path

from ultralytics import YOLO

from src.config import Config, get_project_root


def train(cfg: Config, dataset_yaml: str = None) -> str:
    """
    Train a YOLO model on the converted RPC dataset, then evaluate on the test split.

    Returns the path to the best model weights.
    """
    project_root = get_project_root()
    tc = cfg.training
    wc = cfg.wandb

    if dataset_yaml is None:
        dataset_yaml = str(project_root / cfg.output.root / "dataset.yaml")

    print(f"Model        : {tc.model}")
    print(f"Dataset YAML : {dataset_yaml}")
    print(f"Epochs       : {tc.epochs}")
    print(f"Image size   : {tc.imgsz}")
    print(f"Batch size   : {tc.batch}")
    print(f"Device       : {tc.device}")
    print()

    # --- W&B initialisation (must happen before YOLO.train so Ultralytics picks it up) ---
    wandb_run = None
    if wc.enabled:
        try:
            import wandb

            wandb_run = wandb.init(
                project=wc.project,
                name=wc.run_name,
                config={
                    "model": tc.model,
                    "epochs": tc.epochs,
                    "imgsz": tc.imgsz,
                    "batch": tc.batch,
                    "device": tc.device,
                    "optimizer": tc.optimizer,
                    "lr0": tc.lr0,
                    "lrf": tc.lrf,
                    "patience": tc.patience,
                    "pretrained": tc.pretrained,
                    "dataset_yaml": dataset_yaml,
                },
            )
            print(f"W&B run      : {wandb_run.url}\n")
        except ImportError:
            print("WARNING: wandb not installed — skipping W&B logging. Run: pip install wandb\n")
            wandb_run = None

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

    # --- Evaluate on test split ---
    print("\n--- Running test-set evaluation ---")
    from src.evaluator import evaluate  # local import to avoid circular deps

    metrics = evaluate(cfg, model_path=str(best_weights))

    if wandb_run is not None:
        import wandb

        wandb_run.summary.update({
            "test/map50": metrics["map50"],
            "test/map50_95": metrics["map50_95"],
            "test/precision": metrics["precision"],
            "test/recall": metrics["recall"],
        })
        wandb.finish()

    return str(best_weights)
