"""Track A trainer: linear probe or full fine-tune of a timm backbone.

Class-weighted cross-entropy, mixed precision (CUDA), early stopping on
validation macro-F1, best-checkpoint saving.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from bd_usbench.data.dataset import (
    UltrasoundDataset,
    build_class_list,
    load_classes,
    load_metadata,
    save_classes,
)
from bd_usbench.data.transforms import build_transforms
from bd_usbench.models.classifier import build_classifier
from bd_usbench.utils import ensure_dir, get_logger, resolve_device, save_json, set_seed

logger = get_logger(__name__)


def resolve_class_list(cfg: dict) -> list[str]:
    """Load classes.json if present, otherwise build from the CSV and save it.

    Guarantees every track uses the identical class-index mapping.
    """
    classes_json = Path(cfg["data"]["classes_json"])
    if classes_json.exists():
        return load_classes(classes_json)
    classes = build_class_list(load_metadata(cfg["data"]["metadata_csv"]))
    save_classes(classes, classes_json)
    return classes


def build_dataloaders(
    cfg: dict, classes: list[str]
) -> tuple[DataLoader, DataLoader, UltrasoundDataset]:
    """Train/val dataloaders from the config."""
    data_cfg = cfg["data"]
    image_size = int(data_cfg.get("image_size", 224))
    common = dict(
        metadata_csv=data_cfg["metadata_csv"],
        image_dir=data_cfg["image_dir"],
        domain=data_cfg.get("domain", "all"),
        classes=classes,
        crop_box=data_cfg.get("crop_box"),
    )
    train_ds = UltrasoundDataset(
        split="train",
        transform=build_transforms(image_size, train=True, hflip=bool(data_cfg.get("hflip", False))),
        **common,
    )
    val_ds = UltrasoundDataset(
        split="val", transform=build_transforms(image_size, train=False), **common
    )
    batch_size = int(cfg["train"].get("batch_size", 32))
    num_workers = int(data_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, train_ds


@torch.no_grad()
def evaluate_macro_f1(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Macro-F1 of ``model`` over ``loader``."""
    model.eval()
    preds: list[int] = []
    targets: list[int] = []
    for images, labels in loader:
        logits = model(images.to(device))
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())
    return float(f1_score(targets, preds, average="macro", zero_division=0))


def train_track_a(cfg: dict) -> dict:
    """Run Track A training; returns a summary dict incl. best checkpoint path."""
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    run_dir = ensure_dir(Path(cfg["output"]["dir"]) / cfg["run_name"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    classes = resolve_class_list(cfg)
    train_loader, val_loader, train_ds = build_dataloaders(cfg, classes)

    model = build_classifier(cfg, num_classes=len(classes)).to(device)
    mode = cfg["model"].get("mode", "finetune")

    train_cfg = cfg["train"]
    param_groups = [
        {"params": model.head.parameters(), "lr": float(train_cfg.get("lr_head", 1e-3))}
    ]
    if mode != "linear_probe":
        param_groups.append(
            {
                "params": model.backbone.parameters(),
                "lr": float(train_cfg.get("lr_backbone", 1e-5)),
            }
        )
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=float(train_cfg.get("weight_decay", 0.01))
    )

    weights = (
        train_ds.class_weights().to(device)
        if train_cfg.get("class_weighted_loss", True)
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(train_cfg.get("epochs", 30))
    patience = int(train_cfg.get("early_stopping_patience", 7))
    best_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []
    best_ckpt = ckpt_dir / "best.pt"

    logger.info("Track A: %s | mode=%s | device=%s | %d classes",
                cfg["run_name"], mode, device, len(classes))

    for epoch in range(1, epochs + 1):
        model.train()
        if mode == "linear_probe":
            model.backbone.eval()  # keep norm layers in eval for frozen features
        losses: list[float] = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))

        val_f1 = evaluate_macro_f1(model, val_loader, device)
        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "train_loss": mean_loss, "val_macro_f1": val_f1})
        logger.info("Epoch %d/%d | loss %.4f | val macro-F1 %.4f", epoch, epochs, mean_loss, val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "config": cfg,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                },
                best_ckpt,
            )
            logger.info("New best checkpoint (val macro-F1 %.4f)", val_f1)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d (patience %d)", epoch, patience)
                break

    save_json({"history": history, "best_val_macro_f1": best_f1}, run_dir / "history.json")
    summary = {
        "run_name": cfg["run_name"],
        "best_checkpoint": str(best_ckpt),
        "best_val_macro_f1": best_f1,
        "epochs_run": len(history),
    }
    logger.info("Training done: %s", summary)
    return summary
