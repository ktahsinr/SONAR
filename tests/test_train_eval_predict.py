"""End-to-end smoke test on the synthetic dataset: train Track A (tiny
resnet18, CPU), evaluate, export to TorchScript, and run still-image
prediction — all without network access or real data."""

from __future__ import annotations

from pathlib import Path

import pytest

from bd_usbench.eval.runners import evaluate
from bd_usbench.predict.predictor import StillImagePredictor
from bd_usbench.train.export import export_model
from bd_usbench.train.trainer import train_track_a
from bd_usbench.utils import load_json


@pytest.fixture(scope="module")
def trained(smoke_cfg_module):
    """Train once for all tests in this module."""
    return smoke_cfg_module, train_track_a(smoke_cfg_module)


@pytest.fixture(scope="module")
def smoke_cfg_module(synthetic_data, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("smoke_run")
    # Rebuild the function-scoped fixture config at module scope.
    return {
        "run_name": "smoke_test",
        "seed": 42,
        "device": "cpu",
        "data": {
            "metadata_csv": str(synthetic_data["metadata_csv"]),
            "image_dir": str(synthetic_data["image_dir"]),
            "classes_json": str(tmp_path / "classes.json"),
            "domain": "all",
            "image_size": 64,
            "crop_box": None,
            "hflip": False,
            "num_workers": 0,
        },
        "model": {"backbone": "resnet18", "pretrained": False, "mode": "finetune",
                  "checkpoint_path": None},
        "train": {"epochs": 2, "batch_size": 8, "lr_head": 1e-2, "lr_backbone": 1e-3,
                  "weight_decay": 0.0, "early_stopping_patience": 3, "amp": False,
                  "class_weighted_loss": True},
        "eval": {"batch_size": 8},
        "export": {"format": "torchscript"},
        "output": {"dir": str(tmp_path / "results")},
    }


def test_training_produces_checkpoint_and_history(trained):
    cfg, summary = trained
    ckpt = Path(summary["best_checkpoint"])
    assert ckpt.exists()
    assert 0.0 <= summary["best_val_macro_f1"] <= 1.0
    history = load_json(Path(cfg["output"]["dir"]) / cfg["run_name"] / "history.json")
    assert len(history["history"]) == summary["epochs_run"]
    assert Path(cfg["data"]["classes_json"]).exists()


def test_linear_probe_freezes_backbone(smoke_cfg):
    from bd_usbench.models.classifier import build_classifier

    smoke_cfg["model"]["mode"] = "linear_probe"
    model = build_classifier(smoke_cfg, num_classes=5)
    assert not any(p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_evaluate_writes_results(trained):
    cfg, _ = trained
    metrics = evaluate("a", cfg, split="test", domain="all")
    run_dir = Path(cfg["output"]["dir"]) / cfg["run_name"] / "eval_test_all"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "confusion_matrix.png").exists()
    assert (run_dir / "reliability_diagram.png").exists()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["ece"] is not None
    assert metrics["latency"]["mean_ms"] > 0


def test_evaluate_domain_filter(trained):
    cfg, _ = trained
    metrics = evaluate("a", cfg, split="test", domain="fetal")
    assert metrics["n_samples"] > 0
    assert metrics["domain"] == "fetal"


def test_export_and_still_image_prediction(trained, synthetic_data, tmp_path):
    cfg, summary = trained
    export_dir = tmp_path / "export"
    paths = export_model(summary["best_checkpoint"], export_dir, "torchscript")
    assert paths["model"].exists() and paths["meta"].exists()

    predictor = StillImagePredictor(paths["model"], threshold=0.6)
    image_dir = Path(synthetic_data["image_dir"])

    # Single image
    one_image = sorted(image_dir.iterdir())[0]
    prediction = predictor.predict(one_image)
    assert prediction.organ in predictor.classes
    assert 0.0 <= prediction.confidence <= 1.0
    assert abs(sum(prediction.probs.values()) - 1.0) < 1e-4
    assert prediction.uncertain == (prediction.confidence < 0.6)

    # Folder batch + annotated copies
    predictions = predictor.predict_paths(image_dir)
    assert len(predictions) == len(list(image_dir.glob("*.png")))
    out = StillImagePredictor.save_annotated(predictions[0], tmp_path / "annotated")
    assert out.exists()


def test_compare_script_builds_table(trained, tmp_path):
    cfg, _ = trained
    from bd_usbench.scripts.compare import main as compare_main

    out = tmp_path / "comparison.md"
    rc = compare_main(["--results-dir", cfg["output"]["dir"], "--out", str(out)])
    assert rc == 0
    text = out.read_text()
    assert "macro-F1" in text and "smoke_test" in text


def test_data_report_script(synthetic_data, capsys):
    from bd_usbench.scripts.data_report import main as report_main

    rc = report_main(
        ["--metadata", str(synthetic_data["metadata_csv"]),
         "--image-dir", str(synthetic_data["image_dir"])]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Per-class counts per split" in out
    assert "no patient_id crosses splits" in out
