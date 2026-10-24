"""Backbone registry built on timm.

Registered backbones:
  - ``dinov2_vits14``     : DINOv2 ViT-S/14 (timm, LVD-142M pretraining)
  - ``convnext_tiny``     : ConvNeXt-Tiny (timm, IN-22k ft IN-1k)
  - ``ultrasound_local``  : config slot for an ultrasound-specific checkpoint
                            (e.g. USFM / EchoCare) loaded from a local path
  - ``resnet18``          : small backbone used by smoke tests / CPU runs
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from bd_usbench.utils import get_logger

logger = get_logger(__name__)

BACKBONE_REGISTRY: dict[str, dict] = {
    "dinov2_vits14": {"timm_name": "vit_small_patch14_dinov2.lvd142m", "is_vit": True},
    "convnext_tiny": {"timm_name": "convnext_tiny.fb_in22k_ft_in1k", "is_vit": False},
    "ultrasound_local": {"timm_name": None, "is_vit": True},  # timm_name from config
    "resnet18": {"timm_name": "resnet18", "is_vit": False},
}


def create_backbone(
    name: str,
    pretrained: bool = True,
    image_size: int = 224,
    checkpoint_path: str | None = None,
    timm_name_override: str | None = None,
) -> tuple[nn.Module, int]:
    """Create a headless (num_classes=0) timm backbone.

    Args:
        name: Key in :data:`BACKBONE_REGISTRY`.
        pretrained: Load pretrained weights from timm hub.
        image_size: Input side length (passed as ``img_size`` for ViTs).
        checkpoint_path: Optional local state-dict to load (non-strict). Used
            for ultrasound-specific checkpoints such as USFM / EchoCare.
        timm_name_override: timm architecture name for ``ultrasound_local``.

    Returns:
        (backbone, feature_dim)
    """
    if name not in BACKBONE_REGISTRY:
        raise KeyError(f"Unknown backbone {name!r}. Available: {sorted(BACKBONE_REGISTRY)}")
    entry = BACKBONE_REGISTRY[name]
    timm_name = timm_name_override or entry["timm_name"]
    if timm_name is None:
        raise ValueError(
            f"Backbone {name!r} needs model.timm_name in the config "
            "(the timm architecture matching the local checkpoint)."
        )

    kwargs: dict = {"pretrained": pretrained, "num_classes": 0}
    if entry["is_vit"]:
        kwargs["img_size"] = image_size
    backbone = timm.create_model(timm_name, **kwargs)
    feature_dim = int(backbone.num_features)

    if checkpoint_path:
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {ckpt_file}")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            for key in ("state_dict", "model", "module"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        logger.info(
            "Loaded %s: %d missing / %d unexpected keys",
            ckpt_file.name, len(missing), len(unexpected),
        )

    logger.info("Backbone %s (%s): feature_dim=%d", name, timm_name, feature_dim)
    return backbone, feature_dim
