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
class MetricAugmentationConfig:
    enabled: bool = True
    output_root: str = "metric_dataset"
    img_size: int = 224
    padding_ratio: float = 0.10        # extra context around each bbox before cropping
    copies_per_train_image: int = 5    # augmented copies generated per train crop
    keep_original: bool = True         # also keep the un-augmented crop for train
    seed: int = 42

    # geometric
    horizontal_flip_prob: float = 0.5
    rotate_limit: int = 20
    scale_jitter_min: float = 0.75
    scale_jitter_max: float = 1.25
    perspective_prob: float = 0.3

    # photometric
    color_jitter_prob: float = 0.8
    brightness_contrast_limit: float = 0.3
    hue_shift_limit: int = 15
    sat_shift_limit: int = 25
    blur_prob: float = 0.3
    blur_limit: int = 5
    noise_prob: float = 0.3

    # occlusion
    cutout_prob: float = 0.35
    cutout_max_holes: int = 3
    cutout_max_size_ratio: float = 0.2   # fraction of crop size

    # copy-paste onto real checkout backgrounds (domain-gap bridging)
    copy_paste_prob: float = 0.5
    copy_paste_background_split: str = "val"
    copy_paste_edge_feather: int = 7

    # visualization: saves [orig | aug0 | aug1 | ...] labeled strips for manual inspection
    preview_enabled: bool = True
    preview_samples: int = 12
    preview_dir: str = "_augmentation_preview"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mode: str = "single_class"   # "single_class" | "multi_class"
    classes: List[str] = field(default_factory=lambda: ["product"])
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    metric_augmentation: MetricAugmentationConfig = field(default_factory=MetricAugmentationConfig)


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
    if "mode" in raw:
        cfg.mode = raw["mode"]
    if "classes" in raw:
        cfg.classes = raw["classes"]
    if "training" in raw:
        cfg.training = _dict_to_dataclass(TrainingConfig, raw["training"])
    if "evaluation" in raw:
        cfg.evaluation = _dict_to_dataclass(EvaluationConfig, raw["evaluation"])
    if "wandb" in raw:
        cfg.wandb = _dict_to_dataclass(WandbConfig, raw["wandb"])
    if "metric_augmentation" in raw:
        cfg.metric_augmentation = _dict_to_dataclass(MetricAugmentationConfig, raw["metric_augmentation"])

    return cfg


def get_project_root() -> Path:
    """Return the project root directory (where config.yaml lives)."""
    return Path(__file__).resolve().parent.parent
