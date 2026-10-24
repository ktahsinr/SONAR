"""Backbone + linear head classifier for Track A."""

from __future__ import annotations

import torch
from torch import nn

from bd_usbench.models.registry import create_backbone


class OrganClassifier(nn.Module):
    """timm backbone (headless) followed by a single linear layer."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze (linear probe) or unfreeze (fine-tune) the backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = trainable


def build_classifier(cfg: dict, num_classes: int) -> OrganClassifier:
    """Build an :class:`OrganClassifier` from a config dict.

    Expected keys under ``cfg['model']``: backbone, pretrained, mode,
    optional checkpoint_path and timm_name; ``cfg['data']['image_size']``.
    """
    model_cfg = cfg["model"]
    backbone, feature_dim = create_backbone(
        name=model_cfg["backbone"],
        pretrained=bool(model_cfg.get("pretrained", True)),
        image_size=int(cfg["data"].get("image_size", 224)),
        checkpoint_path=model_cfg.get("checkpoint_path"),
        timm_name_override=model_cfg.get("timm_name"),
    )
    model = OrganClassifier(backbone, feature_dim, num_classes)
    if model_cfg.get("mode", "finetune") == "linear_probe":
        model.set_backbone_trainable(False)
    return model
