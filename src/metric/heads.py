"""Projection head: backbone features -> L2-normalized embedding space."""

import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 256, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)
