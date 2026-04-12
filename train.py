"""Train a YOLO model on the converted RPC dataset."""

import argparse

from src.config import load_config
from src.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on RPC dataset")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default=None, help="Override dataset.yaml path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, dataset_yaml=args.data)


if __name__ == "__main__":
    main()
