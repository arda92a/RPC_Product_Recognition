#!/usr/bin/env python3
"""
Train the DINOv3 + angular-margin metric-learning model for product retrieval.

Usage:
  python train_metric.py --config config.yaml
"""

import argparse

from src.config import load_config
from src.metric.trainer import train_metric_model


def main():
    parser = argparse.ArgumentParser(description="Train the metric-learning product recognition model")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_metric_model(cfg)


if __name__ == "__main__":
    main()
