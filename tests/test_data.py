"""Dataset, transforms, and report tests on the synthetic dataset."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from bd_usbench.data.dataset import (
    UltrasoundDataset,
    build_class_list,
    load_classes,
    load_metadata,
    parse_secondary,
    save_classes,
)
from bd_usbench.data.report import (
    class_split_counts,
    low_count_classes,
    missing_files,
    patient_split_leakage,
)
from bd_usbench.data.synthetic import DEFAULT_CLASSES
from bd_usbench.data.transforms import build_transforms


def test_metadata_loads_and_normalizes(synthetic_data):
    df = load_metadata(synthetic_data["metadata_csv"])
    assert set(df["split"]) == {"train", "val", "test"}
    assert set(df["domain"]) <= {"fetal", "adult"}
    assert (df["primary_organ"] == df["primary_organ"].str.lower()).all()


def test_parse_secondary():
    assert parse_secondary("none") == []
    assert parse_secondary("bowel;spine") == ["bowel", "spine"]
    assert parse_secondary("") == []


def test_class_list_is_sorted_and_stable(synthetic_data, tmp_path):
    df = load_metadata(synthetic_data["metadata_csv"])
    classes = build_class_list(df)
    assert classes == sorted(classes)
    assert set(classes) == set(DEFAULT_CLASSES)
    save_classes(classes, tmp_path / "classes.json")
    assert load_classes(tmp_path / "classes.json") == classes


def test_dataset_split_and_domain_filters(synthetic_data):
    kwargs = dict(
        metadata_csv=synthetic_data["metadata_csv"],
        image_dir=synthetic_data["image_dir"],
        transform=build_transforms(64, train=False),
    )
    train = UltrasoundDataset(split="train", **kwargs)
    val = UltrasoundDataset(split="val", **kwargs)
    test = UltrasoundDataset(split="test", **kwargs)
    full = UltrasoundDataset(**kwargs)
    assert len(train) + len(val) + len(test) == len(full)

    fetal = UltrasoundDataset(split="train", domain="fetal", **kwargs)
    assert all(fetal.df["domain"] == "fetal")
    assert len(fetal) < len(train)

    # Class indices must be identical regardless of filtering.
    assert fetal.class_to_idx == full.class_to_idx


def test_dataset_getitem_shapes_and_labels(synthetic_data):
    ds = UltrasoundDataset(
        metadata_csv=synthetic_data["metadata_csv"],
        image_dir=synthetic_data["image_dir"],
        split="train",
        transform=build_transforms(64, train=True),
    )
    image, label = ds[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert 0 <= label < len(ds.classes)
    assert ds.image_path(0).exists()


def test_crop_box_applied(synthetic_data):
    ds = UltrasoundDataset(
        metadata_csv=synthetic_data["metadata_csv"],
        image_dir=synthetic_data["image_dir"],
        split="train",
        transform=build_transforms(32, train=False),
        crop_box=(8, 8, 40, 40),
    )
    image, _ = ds[0]
    assert image.shape == (3, 32, 32)


def test_class_weights_positive(synthetic_data):
    ds = UltrasoundDataset(
        metadata_csv=synthetic_data["metadata_csv"],
        image_dir=synthetic_data["image_dir"],
        split="train",
    )
    weights = ds.class_weights()
    assert (weights > 0).all()
    assert len(weights) == len(ds.classes)


def test_report_counts_and_checks(synthetic_data):
    df = load_metadata(synthetic_data["metadata_csv"])
    counts = class_split_counts(df)
    assert set(counts.columns) == {"train", "val", "test", "total"}
    assert counts["total"].sum() == len(df)
    # Synthetic classes have 6 train images each -> all flagged below 30.
    assert len(low_count_classes(df, min_train=30)) == len(DEFAULT_CLASSES)
    assert low_count_classes(df, min_train=1) == []
    assert patient_split_leakage(df) == []
    assert missing_files(df, synthetic_data["image_dir"]) == []


def test_leakage_detected_on_bad_split():
    df = pd.DataFrame(
        {"patient_id": ["p1", "p1", "p2"], "split": ["train", "test", "val"]}
    )
    assert patient_split_leakage(df) == ["p1"]


def test_unknown_split_rejected(synthetic_data):
    with pytest.raises(ValueError):
        UltrasoundDataset(
            metadata_csv=synthetic_data["metadata_csv"],
            image_dir=synthetic_data["image_dir"],
            split="bogus",
        )
