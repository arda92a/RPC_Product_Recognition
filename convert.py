"""Convert the RPC dataset from COCO format to YOLO single-class format."""

import argparse
from pathlib import Path

from src.config import load_config
from src.converter import convert_dataset, convert_one_split


def main():
    parser = argparse.ArgumentParser(description="Convert RPC dataset to YOLO format")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None,
                         help="Only re-convert this single split (leaves the other splits' "
                              "images/labels dirs untouched) — use this to refresh val2019_clean "
                              "without wiping the purged/synthetic train split")
    parser.add_argument("--output-root", type=str, default=None,
                         help="Override output.root, e.g. to target yolo_dataset_rpc directly")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.split:
        output_root = Path(args.output_root) if args.output_root else None
        convert_one_split(cfg, args.split, output_root=output_root, use_symlinks=not args.copy)
    else:
        convert_dataset(cfg, use_symlinks=not args.copy)


if __name__ == "__main__":
    main()
