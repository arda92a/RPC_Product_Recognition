"""Convert the RPC dataset from COCO format to YOLO single-class format."""

import argparse

from src.config import load_config
from src.converter import convert_dataset


def main():
    parser = argparse.ArgumentParser(description="Convert RPC dataset to YOLO format")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    args = parser.parse_args()

    cfg = load_config(args.config)
    convert_dataset(cfg, use_symlinks=not args.copy)


if __name__ == "__main__":
    main()
