"""
Margin-based classification losses for metric learning (ArcFace family).

Each loss keeps a learnable per-class weight matrix, compares L2-normalized
embeddings against L2-normalized class prototypes via cosine similarity,
injects an angular/cosine margin at the ground-truth class, then scales
before feeding into standard cross-entropy. All losses return (loss, logits)
so the trainer can also report classification accuracy for free.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BaseMarginLoss(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, scale: float = 64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(embeddings), F.normalize(self.weight))


class CosFaceLoss(_BaseMarginLoss):
    """AM-Softmax / CosFace: subtract a fixed margin from the target cosine."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.35, scale: float = 64.0):
        super().__init__(embed_dim, num_classes, scale)
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        cos_theta = self.cosine(embeddings).clamp(-1 + 1e-7, 1 - 1e-7)
        target = cos_theta.gather(1, labels.view(-1, 1)) - self.margin
        logits = cos_theta.scatter(1, labels.view(-1, 1), target)
        logits = logits * self.scale
        return F.cross_entropy(logits, labels), logits


class ArcFaceLoss(_BaseMarginLoss):
    """Additive angular margin: margin is added in angle space at the target class."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        super().__init__(embed_dim, num_classes, scale)
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # stabilizes the margin past pi - m, where cos() stops being monotonic
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        cos_theta = self.cosine(embeddings).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=0))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.scale
        return F.cross_entropy(logits, labels), logits


class SubcenterArcFaceLoss(nn.Module):
    """ArcFace with K sub-centers per class (Deng et al. 2020) — more robust to
    multi-modal/noisy classes (e.g. very different product poses) since only the
    best-matching sub-center per class has to be close, not a single prototype."""

    def __init__(self, embed_dim: int, num_classes: int, k: int = 3, margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.weight = nn.Parameter(torch.empty(num_classes * k, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        cos_all = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        cos_theta, _ = cos_all.view(-1, self.num_classes, self.k).max(dim=2)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=0))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)
        logits = (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.scale
        return F.cross_entropy(logits, labels), logits


class CircleLoss(_BaseMarginLoss):
    """Class-level Circle Loss (Sun et al. 2020): applies an adaptive (self-paced)
    weight to positive/negative cosine similarities instead of a fixed margin,
    which tends to be less sensitive to margin hyperparameter choice."""

    def __init__(self, embed_dim: int, num_classes: int, margin: float = 0.25, scale: float = 256.0):
        super().__init__(embed_dim, num_classes, scale)
        self.margin = margin
        self.op = 1 + margin
        self.on = -margin
        self.delta_p = 1 - margin
        self.delta_n = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        cos_theta = self.cosine(embeddings)
        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1.0)

        alpha_p = (self.op - cos_theta).clamp(min=0).detach()
        alpha_n = (cos_theta - self.on).clamp(min=0).detach()

        logits_p = alpha_p * (cos_theta - self.delta_p) * self.scale
        logits_n = alpha_n * (cos_theta - self.delta_n) * self.scale
        logits = one_hot * logits_p + (1.0 - one_hot) * logits_n
        return F.cross_entropy(logits, labels), logits


def build_loss(name: str, embed_dim: int, num_classes: int, margin: float, scale: float,
               subcenters: int = 3) -> nn.Module:
    name = name.lower()
    if name == "cosface":
        return CosFaceLoss(embed_dim, num_classes, margin=margin, scale=scale)
    if name == "arcface":
        return ArcFaceLoss(embed_dim, num_classes, margin=margin, scale=scale)
    if name == "subcenter_arcface":
        return SubcenterArcFaceLoss(embed_dim, num_classes, k=subcenters, margin=margin, scale=scale)
    if name == "circle":
        return CircleLoss(embed_dim, num_classes, margin=margin, scale=scale)
    raise ValueError(f"Unknown metric loss: {name}")
