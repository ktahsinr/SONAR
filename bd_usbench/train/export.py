"""Export a trained Track A checkpoint to TorchScript (default) or ONNX for
CPU inference, with an export_meta.json describing preprocessing."""

from __future__ import annotations

from pathlib import Path

import torch

from bd_usbench.models.classifier import build_classifier
from bd_usbench.utils import IMAGENET_MEAN, IMAGENET_STD, ensure_dir, get_logger, save_json

logger = get_logger(__name__)


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load a Track A training checkpoint (weights + classes + config)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_state", "classes", "config"):
        if key not in ckpt:
            raise ValueError(f"Checkpoint {checkpoint_path} is missing key {key!r}")
    return ckpt


def export_model(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    export_format: str = "torchscript",
) -> dict[str, Path]:
    """Export a checkpoint; returns paths of the model file and its metadata.

    TorchScript is the default (no extra runtime dependency); ONNX export is
    also supported (requires the ``onnx`` package).
    """
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    classes: list[str] = ckpt["classes"]
    image_size = int(cfg["data"].get("image_size", 224))

    model = build_classifier(cfg, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    output_dir = ensure_dir(output_dir)
    example = torch.randn(1, 3, image_size, image_size)

    if export_format == "torchscript":
        model_path = Path(output_dir) / "model.torchscript.pt"
        traced = torch.jit.trace(model, example)
        traced.save(str(model_path))
    elif export_format == "onnx":
        model_path = Path(output_dir) / "model.onnx"
        torch.onnx.export(
            model,
            example,
            str(model_path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    else:
        raise ValueError(f"Unknown export format {export_format!r} (torchscript|onnx)")

    meta_path = Path(output_dir) / "export_meta.json"
    save_json(
        {
            "format": export_format,
            "classes": classes,
            "image_size": image_size,
            "crop_box": cfg["data"].get("crop_box"),
            "normalize_mean": list(IMAGENET_MEAN),
            "normalize_std": list(IMAGENET_STD),
            "source_checkpoint": str(checkpoint_path),
            "run_name": cfg.get("run_name"),
            "val_macro_f1": ckpt.get("val_macro_f1"),
        },
        meta_path,
    )
    logger.info("Exported %s model to %s", export_format, model_path)
    return {"model": model_path, "meta": meta_path}
