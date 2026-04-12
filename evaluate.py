"""Evaluate a trained YOLO model on the RPC test set."""

import argparse

from src.config import load_config
from src.evaluator import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO on RPC test set")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Override model weights path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, model_path=args.model)


if __name__ == "__main__":
    main()
