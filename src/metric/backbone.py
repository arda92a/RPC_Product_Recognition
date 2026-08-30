"""
DINOv3 backbone wrapper with staged fine-tuning support (frozen linear-probe
stage, then partial unfreeze of the last N transformer blocks).

Loading is intentionally flexible since DINOv3 can be obtained either via
torch.hub (facebookresearch/dinov3) or the HuggingFace transformers hub —
verify the exact model name / id available on your server before the first
real run (a quick `python -c "from src.metric.backbone import DinoV3Backbone;
b = DinoV3Backbone(); print(b.feature_dim)"` is enough to sanity-check).

When a local checkpoint_path is given (e.g. a manually downloaded
dinov3_vits16_pretrain_lvd1689m-*.pth), it's loaded into the architecture
instead of letting torch.hub fetch pretrained weights over the network —
torch.hub still needs network access once to fetch the repo's model code.
"""

import torch
import torch.nn as nn


class DinoV3Backbone(nn.Module):
    def __init__(self, model_name: str = "dinov3_vits16", source: str = "torchhub",
                 hf_model_id: str = None, checkpoint_path: str = None):
        super().__init__()
        self.source = source

        if source == "torchhub":
            self.model = self._load_torchhub(model_name, checkpoint_path)
        elif source == "hf":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(hf_model_id or model_name)
        else:
            raise ValueError(f"Unknown backbone source: {source}")

        self.feature_dim = self._infer_feature_dim()

    def _load_torchhub(self, model_name: str, checkpoint_path: str):
        if not checkpoint_path:
            return torch.hub.load("facebookresearch/dinov3", model_name)
        try:
            # most dinov3/dinov2-style hubconfs accept a local `weights` override
            return torch.hub.load("facebookresearch/dinov3", model_name, weights=checkpoint_path)
        except TypeError:
            model = torch.hub.load("facebookresearch/dinov3", model_name, pretrained=False)
            self._load_local_state_dict(model, checkpoint_path)
            return model

    @staticmethod
    def _load_local_state_dict(model: nn.Module, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in ("model", "state_dict", "teacher"):
            if isinstance(state_dict, dict) and key in state_dict:
                state_dict = state_dict[key]
        state_dict = {k.replace("backbone.", "").replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[DinoV3Backbone] loaded '{checkpoint_path}' with "
                  f"{len(missing)} missing / {len(unexpected)} unexpected keys")

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
