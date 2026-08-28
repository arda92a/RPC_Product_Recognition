"""Combines a backbone + embedding head into a single embedding extractor."""

import torch.nn as nn

from src.metric.backbone import DinoV3Backbone
from src.metric.heads import EmbeddingHead


class MetricModel(nn.Module):
    def __init__(self, backbone: DinoV3Backbone, embed_dim: int = 256,
                 hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        self.head = EmbeddingHead(backbone.feature_dim, embed_dim=embed_dim,
                                   hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x):
        return self.head(self.backbone(x))
