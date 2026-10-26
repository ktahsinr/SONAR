"""Per-track prediction runners and the shared evaluate() entry point.

``evaluate(track, cfg, split, domain)`` writes to results/<run_name>/eval_<split>_<domain>/:
metrics.json, summary.md, confusion_matrix.png, reliability_diagram.png (when
real probabilities exist), and unparsed.csv for generative VLM failures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from bd_usbench.data.dataset import UltrasoundDataset, load_classes
from bd_usbench.data.transforms import build_transforms
from bd_usbench.eval.latency import measure_latency
from bd_usbench.eval.metrics import (
    compute_classification_metrics,
    expected_calibration_error,
    plot_confusion_matrix,
    plot_reliability_diagram,
)
from bd_usbench.models.classifier import build_classifier
from bd_usbench.utils import ensure_dir, get_logger, resolve_device, save_json, set_seed

logger = get_logger(__name__)


def _dataset(cfg: dict, classes: list[str], split: str, domain: str, with_transform: bool) -> UltrasoundDataset:
    data_cfg = cfg["data"]
    transform = (
        build_transforms(int(data_cfg.get("image_size", 224)), train=False)
        if with_transform
        else None
    )
    return UltrasoundDataset(
        metadata_csv=data_cfg["metadata_csv"],
        image_dir=data_cfg["image_dir"],
        split=split,
        domain=domain,
        classes=classes,
        transform=transform,
        crop_box=data_cfg.get("crop_box"),
        min_train_images=int(data_cfg.get("min_train_images", 5)),
    )


def predictions_track_a(
    cfg: dict, classes: list[str], split: str, domain: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict]:
    """Run the trained supervised classifier; returns (y_true, y_pred, probs, latency)."""
    device = resolve_device(cfg.get("device", "auto"))
    checkpoint = cfg.get("eval", {}).get("checkpoint") or str(
        Path(cfg["output"]["dir"]) / cfg["run_name"] / "checkpoints" / "best.pt"
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if ckpt["classes"] != classes:
        raise ValueError(
            f"Checkpoint class list differs from classes.json ({checkpoint}). "
            "Delete stale artifacts or point eval.checkpoint at the right run."
        )
    model = build_classifier(cfg, num_classes=len(classes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = _dataset(cfg, classes, split, domain, with_transform=True)
    loader = DataLoader(ds, batch_size=int(cfg.get("eval", {}).get("batch_size", 32)))
    all_probs: list[np.ndarray] = []
    y_true: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            probs = model(images.to(device)).softmax(dim=1).cpu().numpy()
            all_probs.append(probs)
            y_true.extend(labels.tolist())
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, len(classes)))
    y_pred = probs.argmax(axis=1) if len(probs) else np.array([], dtype=int)

    sample, _ = ds[0]
    sample = sample.unsqueeze(0).to(device)
    with torch.no_grad():
        latency = measure_latency(lambda: model(sample))
    return np.array(y_true), y_pred, probs, latency


def predictions_track_b(
    cfg: dict, classes: list[str], split: str, domain: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict, list[dict]]:
    """Zero-shot VLM predictions. CLIP yields real probabilities; generative
    Qwen yields hard predictions (probs=None) plus a log of unparseable outputs."""
    ds = _dataset(cfg, classes, split, domain, with_transform=False)
    y_true = np.array([ds.label_index(i) for i in range(len(ds))])
    vlm_type = cfg["model"].get("type", "clip")
    unparsed: list[dict] = []

    if vlm_type == "clip":
        from bd_usbench.models.vlm.clip_zeroshot import ClipZeroShot

        device = resolve_device(cfg.get("device", "auto"))
        clip = ClipZeroShot(
            classes, model_name=cfg["model"].get("clip_model", "biomedclip"), device=device
        )
        probs = np.stack([clip.predict(ds.image_path(i)).numpy() for i in range(len(ds))])
        y_pred = probs.argmax(axis=1)
        latency = measure_latency(lambda: clip.predict(ds.image_path(0)), n_runs=10)
        return y_true, y_pred, probs, latency, unparsed

    if vlm_type == "qwen":
        from bd_usbench.models.vlm.qwen import QwenVLM

        vlm = QwenVLM(
            model_name=cfg["model"].get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct"),
            load_in_4bit=bool(cfg["model"].get("load_in_4bit", True)),
            adapter_path=cfg["model"].get("adapter_path"),
        )
        y_pred = np.full(len(ds), -1, dtype=int)
        for i in range(len(ds)):
            pred_class, raw = vlm.classify(ds.image_path(i), classes)
            if pred_class is None:
                unparsed.append({"filename": str(ds.df.iloc[i]["filename"]), "raw_output": raw})
                logger.warning("Unparseable VLM output for %s: %r", ds.image_path(i).name, raw)
            else:
                y_pred[i] = classes.index(pred_class)
        latency = measure_latency(
            lambda: vlm.classify(ds.image_path(0), classes), n_warmup=1, n_runs=5
        )
        return y_true, y_pred, None, latency, unparsed

    raise ValueError(f"Unknown VLM type {vlm_type!r} (clip|qwen)")


def predictions_track_c(
    cfg: dict, classes: list[str], split: str, domain: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict, list[dict]]:
    """Track C = Track B's Qwen path with the trained LoRA adapter attached."""
    cfg = dict(cfg)
    model_cfg = dict(cfg["model"])
    model_cfg["type"] = "qwen"
    model_cfg.setdefault(
        "adapter_path", str(Path(cfg["output"]["dir"]) / cfg["run_name"] / "adapter")
    )
    cfg["model"] = model_cfg
    return predictions_track_b(cfg, classes, split, domain)


def evaluate(track: str, cfg: dict, split: str = "test", domain: str = "all") -> dict:
    """Shared evaluation entry point for all tracks; writes the results folder."""
    set_seed(int(cfg.get("seed", 42)))
    classes = load_classes(cfg["data"]["classes_json"])
    out_dir = ensure_dir(
        Path(cfg["output"]["dir"]) / cfg["run_name"] / f"eval_{split}_{domain}"
    )

    unparsed: list[dict] = []
    if track == "a":
        y_true, y_pred, probs, latency = predictions_track_a(cfg, classes, split, domain)
    elif track == "b":
        y_true, y_pred, probs, latency, unparsed = predictions_track_b(cfg, classes, split, domain)
    elif track == "c":
        y_true, y_pred, probs, latency, unparsed = predictions_track_c(cfg, classes, split, domain)
    else:
        raise ValueError(f"Unknown track {track!r} (a|b|c)")

    metrics = compute_classification_metrics(y_true, y_pred, classes)
    metrics.update(
        {
            "track": track,
            "run_name": cfg["run_name"],
            "split": split,
            "domain": domain,
            "latency": latency,
            "device": str(resolve_device(cfg.get("device", "auto"))),
        }
    )
    plot_confusion_matrix(y_true, y_pred, classes, out_dir / "confusion_matrix.png")
    if probs is not None and len(probs):
        metrics["ece"] = expected_calibration_error(y_true, probs)
        plot_reliability_diagram(y_true, probs, out_dir / "reliability_diagram.png")
    else:
        metrics["ece"] = None  # generative VLM: no calibrated probabilities

    if unparsed:
        pd.DataFrame(unparsed).to_csv(out_dir / "unparsed.csv", index=False)

    save_json(metrics, out_dir / "metrics.json")
    _write_summary_md(metrics, out_dir / "summary.md")
    logger.info(
        "Eval [%s/%s/%s]: acc=%.4f macro-F1=%.4f -> %s",
        track, split, domain, metrics["accuracy"], metrics["macro_f1"], out_dir,
    )
    return metrics


def _write_summary_md(metrics: dict, path: Path) -> None:
    lines = [
        f"# {metrics['run_name']} — track {metrics['track'].upper()} "
        f"({metrics['split']}, domain={metrics['domain']})",
        "",
        f"- Accuracy: **{metrics['accuracy']:.4f}**",
        f"- Macro-F1: **{metrics['macro_f1']:.4f}**",
        f"- ECE: {metrics['ece'] if metrics['ece'] is not None else 'n/a (no probabilities)'}",
        f"- Mean latency: {metrics['latency']['mean_ms']} ms/image on {metrics['device']}",
        f"- Samples: {metrics['n_samples']} (unparsed: {metrics['n_unparsed']})",
        "",
        "| class | precision | recall | f1 | support |",
        "|---|---|---|---|---|",
    ]
    for cls, m in metrics["per_class"].items():
        lines.append(
            f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
