from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


@dataclass
class DatasetConfig:
    root: str = "RPC_dataset"
    annotations: dict = field(default_factory=lambda: {
        "train": "instances_train2019.json",
        "val": "instances_val2019.json",
        "test": "instances_test2019.json",
    })
    images: dict = field(default_factory=lambda: {
        "train": "train2019",
        "val": "val2019",
        "test": "test2019",
    })


@dataclass
class OutputConfig:
    root: str = "yolo_dataset"


@dataclass
class TrainingConfig:
    model: str = "yolo11n.pt"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "0"
    workers: int = 8
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    patience: int = 50
    project: str = "runs"
    name: str = "rpc_product_detection"
    pretrained: bool = True
    resume: bool = False
    exist_ok: bool = False


@dataclass
class EvaluationConfig:
    model: str = "runs/rpc_product_detection/weights/best.pt"
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    device: str = "0"
    split: str = "test"


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "rpc-product-detection"
    run_name: str = "rpc_yolo_run"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    classes: List[str] = field(default_factory=lambda: ["product"])
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _dict_to_dataclass(cls, data: dict):
    """Map a flat dictionary to a dataclass, ignoring unknown keys."""
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in valid_keys})


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file and return a Config object."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config()

    if "dataset" in raw:
        cfg.dataset = _dict_to_dataclass(DatasetConfig, raw["dataset"])
    if "output" in raw:
        cfg.output = _dict_to_dataclass(OutputConfig, raw["output"])
    if "classes" in raw:
        cfg.classes = raw["classes"]
    if "training" in raw:
        cfg.training = _dict_to_dataclass(TrainingConfig, raw["training"])
    if "evaluation" in raw:
        cfg.evaluation = _dict_to_dataclass(EvaluationConfig, raw["evaluation"])
    if "wandb" in raw:
        cfg.wandb = _dict_to_dataclass(WandbConfig, raw["wandb"])

    return cfg


def get_project_root() -> Path:
    """Return the project root directory (where config.yaml lives)."""
    return Path(__file__).resolve().parent.parent
