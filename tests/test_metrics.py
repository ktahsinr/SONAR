"""Metric computation, calibration, and plotting tests (no model needed)."""

from __future__ import annotations

import numpy as np

from bd_usbench.eval.metrics import (
    compute_classification_metrics,
    expected_calibration_error,
    plot_confusion_matrix,
    plot_reliability_diagram,
)


CLASSES = ["bladder", "heart", "liver"]


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    metrics = compute_classification_metrics(y, y, CLASSES)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["per_class"]["liver"]["recall"] == 1.0
    assert metrics["n_unparsed"] == 0


def test_unparsed_counts_as_error():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, -1, 2])  # one unparseable VLM answer
    metrics = compute_classification_metrics(y_true, y_pred, CLASSES)
    assert metrics["n_unparsed"] == 1
    assert metrics["accuracy"] < 1.0
    assert metrics["per_class"]["heart"]["recall"] == 0.0


def test_ece_bounds():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=200)
    y_true = rng.integers(0, 3, size=200)
    ece = expected_calibration_error(y_true, probs)
    assert 0.0 <= ece <= 1.0

    # Perfectly confident and correct -> ECE 0.
    y = np.array([0, 1, 2])
    one_hot = np.eye(3)[y]
    assert expected_calibration_error(y, one_hot) == 0.0


def test_plots_written(tmp_path):
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, size=50)
    probs = rng.dirichlet(np.ones(3), size=50)
    y_pred = probs.argmax(axis=1)
    y_pred[0] = -1  # include an unparsed prediction

    cm_path = tmp_path / "cm.png"
    rel_path = tmp_path / "rel.png"
    plot_confusion_matrix(y_true, y_pred, CLASSES, cm_path)
    plot_reliability_diagram(y_true, probs, rel_path)
    assert cm_path.exists() and cm_path.stat().st_size > 0
    assert rel_path.exists() and rel_path.stat().st_size > 0
