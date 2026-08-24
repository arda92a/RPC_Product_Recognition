"""Build the augmented per-class crop dataset used for metric-learning training."""

import argparse

from src.augmentor import build_metric_dataset
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Build augmented RPC metric-learning crop dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--splits", nargs="+", default=None, choices=["train", "val", "test"],
        help="Which splits to process (default: all)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_metric_dataset(cfg, splits=args.splits)


if __name__ == "__main__":
    main()
