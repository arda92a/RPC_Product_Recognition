"""
DINOv3 backbone wrapper with staged fine-tuning support (frozen linear-probe
stage, then partial unfreeze of the last N transformer blocks).

Loading is intentionally flexible since DINOv3 can be obtained either via
torch.hub (facebookresearch/dinov3) or the HuggingFace transformers hub —
verify the exact model name / id available on your server before the first
real run (a quick `python -c "from src.metric.backbone import DinoV3Backbone;
b = DinoV3Backbone(); print(b.feature_dim)"` is enough to sanity-check).
"""

import torch
import torch.nn as nn


class DinoV3Backbone(nn.Module):
    def __init__(self, model_name: str = "dinov3_vits16", source: str = "torchhub",
                 hf_model_id: str = None):
        super().__init__()
        self.source = source

        if source == "torchhub":
            self.model = torch.hub.load("facebookresearch/dinov3", model_name)
        elif source == "hf":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(hf_model_id or model_name)
        else:
            raise ValueError(f"Unknown backbone source: {source}")

        self.feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self) -> int:
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            dim = self._forward_features(torch.zeros(1, 3, 224, 224)).shape[-1]
        self.model.train(was_training)
        return dim

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.source == "torchhub":
            return self.model(x)  # dinov3 hub entrypoints return pooled CLS features
        out = self.model(pixel_values=x)
        return out.last_hidden_state[:, 0]  # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_features(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int):
        """Unfreeze only the last n transformer blocks (+ final norm); everything
        else (patch embed, early blocks) stays frozen for stage-2 fine-tuning."""
        blocks = self._get_blocks()
        if not blocks:
            for p in self.parameters():
                p.requires_grad = True
            return
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
        for name, module in self.model.named_modules():
            if name.split(".")[-1] == "norm":
                for p in module.parameters(recurse=False):
                    p.requires_grad = True

    def _get_blocks(self):
        for attr in ("blocks", "layer", "layers"):
            blocks = getattr(self.model, attr, None)
            if blocks is not None:
                return list(blocks)
            encoder = getattr(self.model, "encoder", None)
            if encoder is not None and hasattr(encoder, attr):
                return list(getattr(encoder, attr))
        return None
