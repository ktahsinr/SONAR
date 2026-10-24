"""Shared fixtures: a session-scoped synthetic dataset so no real data is needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from bd_usbench.data.synthetic import make_synthetic_dataset


@pytest.fixture(scope="session")
def synthetic_data(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate a small synthetic dataset once per test session."""
    root = tmp_path_factory.mktemp("synthetic")
    return make_synthetic_dataset(root, images_per_patient=3, image_size=64, seed=7)


@pytest.fixture()
def smoke_cfg(synthetic_data: dict[str, Path], tmp_path: Path) -> dict:
    """Minimal CPU training config over the synthetic dataset."""
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
        "model": {
            "backbone": "resnet18",
            "pretrained": False,
            "mode": "finetune",
            "checkpoint_path": None,
        },
        "train": {
            "epochs": 2,
            "batch_size": 8,
            "lr_head": 1e-2,
            "lr_backbone": 1e-3,
            "weight_decay": 0.0,
            "early_stopping_patience": 3,
            "amp": False,
            "class_weighted_loss": True,
        },
        "eval": {"batch_size": 8},
        "export": {"format": "torchscript"},
        "output": {"dir": str(tmp_path / "results")},
    }
