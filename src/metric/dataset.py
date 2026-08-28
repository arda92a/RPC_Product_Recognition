"""
Metric-learning crop dataset: reads metric_dataset/{split}/<class_name>/*.jpg
produced by src/augmentor.py (ImageFolder-style). Train crops are already
augmented offline (multiple copies per source image), so no additional
online augmentation is applied here — just resize + normalize.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MetricCropDataset(Dataset):
    def __init__(self, root: Path, split: str, img_size: int = 224,
                 class_to_idx: Dict[str, int] = None, train: bool = False):
        self.split_dir = Path(root) / split
        self.samples: List[Tuple[Path, int]] = []

        class_names = sorted(d.name for d in self.split_dir.iterdir() if d.is_dir())
        if class_to_idx is None:
            class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.class_to_idx = class_to_idx
        self.classes = [None] * len(class_to_idx)
        for name, idx in class_to_idx.items():
            self.classes[idx] = name

        for name in class_names:
            if name not in class_to_idx:
                continue
            idx = class_to_idx[name]
            for img_path in (self.split_dir / name).glob("*.jpg"):
                self.samples.append((img_path, idx))

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), label


def build_metric_datasets(root: Path, img_size: int):
    """Build train/val/test datasets sharing one consistent class_to_idx (derived
    from train) so labels line up across splits for retrieval evaluation."""
    train_ds = MetricCropDataset(root, "train", img_size=img_size, train=True)
    val_ds = MetricCropDataset(root, "val", img_size=img_size,
                                class_to_idx=train_ds.class_to_idx, train=False)
    test_ds = MetricCropDataset(root, "test", img_size=img_size,
                                 class_to_idx=train_ds.class_to_idx, train=False)
    return train_ds, val_ds, test_ds
