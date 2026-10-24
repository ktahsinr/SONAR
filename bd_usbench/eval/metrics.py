"""Classification metrics, confusion matrix plot, ECE and reliability diagram.

Predictions may contain -1 for "unparseable" VLM outputs; those count as
errors and appear as an extra 'unparsed' column in the confusion matrix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict:
    """Accuracy, macro-F1 and per-class precision/recall/F1/support."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(classes)))
    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        cls: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, cls in enumerate(classes)
    }
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "n_samples": int(len(y_true)),
        "n_unparsed": int((y_pred == -1).sum()),
    }


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15
) -> float:
    """ECE with equal-width confidence bins over the max predicted probability."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], out_path: str | Path
) -> None:
    """Save a row-annotated confusion matrix PNG (extra column for unparsed)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    col_names = list(classes)
    y_pred_plot = y_pred.copy()
    if (y_pred == -1).any():
        col_names.append("unparsed")
        y_pred_plot[y_pred_plot == -1] = len(classes)
    cm = confusion_matrix(
        y_true, y_pred_plot, labels=list(range(len(col_names)))
    )[: len(classes)]  # rows: true classes only

    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(col_names)), max(5, 0.6 * len(classes))))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(col_names)), col_names, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram(
    y_true: np.ndarray, probs: np.ndarray, out_path: str | Path, n_bins: int = 15
) -> None:
    """Save a reliability diagram (accuracy vs confidence per bin) PNG."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, fractions = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        accs.append(correct[mask].mean())
        fractions.append(mask.mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.bar(centers, accs, width=1.0 / n_bins * 0.9, alpha=0.7, label="accuracy")
    ax.plot(centers, centers, alpha=0)  # keep limits sane
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE = {expected_calibration_error(y_true, probs):.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
