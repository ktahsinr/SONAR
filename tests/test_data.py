"""Dataset, transforms, and report tests on the synthetic dataset."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from bd_usbench.data.dataset import (
    OTHER_CLASS,
    UltrasoundDataset,
    build_class_list,
    collapse_rare_classes,
    load_classes,
    load_metadata,
    parse_secondary,
    save_classes,
)
from bd_usbench.data.report import (
    build_report,
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


def test_filenames_kept_verbatim_and_none_not_nan(tmp_path):
    # Mixed-case filenames (P036_PAN_GB.JPG.JPG etc.) must be opened exactly as
    # written on case-sensitive filesystems, and the literal string "none" must
    # survive CSV parsing instead of becoming NaN.
    row = {
        "filename": "P036_PAN_GB.JPG.JPG",
        "patient_id": "P036",
        "primary_organ": "Gallbladder",
        "secondary_organ": "none",
        "finding": "none",
        "plane": "transverse",
        "machine": "m1",
        "quality": "good",
        "confidence": "high",
        "labeled_by": "a",
        "reviewed_by": "none",
        "split": "train",
        "domain": "adult",
    }
    csv = tmp_path / "metadata.csv"
    pd.DataFrame([row]).to_csv(csv, index=False)
    df = load_metadata(csv)
    assert df.loc[0, "filename"] == "P036_PAN_GB.JPG.JPG"  # case preserved
    assert df.loc[0, "primary_organ"] == "gallbladder"  # labels still lowered
    assert df.loc[0, "secondary_organ"] == "none"
    assert parse_secondary(df.loc[0, "secondary_organ"]) == []


@pytest.fixture()
def rare_class_csv(synthetic_data, tmp_path):
    """Synthetic metadata with 'fetal_head' reduced to 2 train images."""
    df = pd.read_csv(synthetic_data["metadata_csv"], dtype=str, keep_default_na=False)
    mask = (df["primary_organ"] == "fetal_head") & (df["split"] == "train")
    df = df.drop(df[mask].index[2:])
    csv = tmp_path / "metadata.csv"
    df.to_csv(csv, index=False)
    return csv


def test_collapse_rare_classes(rare_class_csv):
    df = load_metadata(rare_class_csv)
    collapsed_df, collapsed = collapse_rare_classes(df, min_train_images=5)
    assert collapsed == ["fetal_head"]
    assert "fetal_head" not in set(collapsed_df["primary_organ"])
    # val/test rows of the rare class are remapped too.
    assert (collapsed_df["primary_organ"] == OTHER_CLASS).sum() == (
        df["primary_organ"] == "fetal_head"
    ).sum()
    # Threshold 0 disables collapsing entirely.
    same_df, none_collapsed = collapse_rare_classes(df, min_train_images=0)
    assert none_collapsed == [] and same_df is df

    classes = build_class_list(df, min_train_images=5)
    assert classes == sorted(set(DEFAULT_CLASSES) - {"fetal_head"} | {OTHER_CLASS})
    assert "fetal_head" in build_class_list(df)


def test_dataset_collapses_consistently_across_splits(synthetic_data, rare_class_csv):
    kwargs = dict(
        metadata_csv=rare_class_csv,
        image_dir=synthetic_data["image_dir"],
        min_train_images=5,
    )
    train = UltrasoundDataset(split="train", **kwargs)
    val = UltrasoundDataset(split="val", **kwargs)
    test = UltrasoundDataset(split="test", **kwargs)
    assert train.collapsed_classes == ["fetal_head"]
    assert train.classes == val.classes == test.classes
    assert OTHER_CLASS in train.classes and "fetal_head" not in train.classes
    assert train.class_to_idx == val.class_to_idx == test.class_to_idx
    # The rare class had a full patient in val/test: those samples become "other".
    assert (val.df["primary_organ"] == OTHER_CLASS).sum() == 3
    assert (test.df["primary_organ"] == OTHER_CLASS).sum() == 3
    # An explicit (reduced) class list from classes.json is accepted unchanged.
    explicit = UltrasoundDataset(split="val", classes=train.classes, **kwargs)
    assert explicit.class_to_idx == train.class_to_idx


def test_report_lists_collapsed_classes(synthetic_data, rare_class_csv):
    report = build_report(
        rare_class_csv, image_dir=synthetic_data["image_dir"], min_train_images=5
    )
    assert f"collapsed into '{OTHER_CLASS}'" in report
    assert "fetal_head: 2 train images" in report
    report_ok = build_report(synthetic_data["metadata_csv"], min_train_images=5)
    assert "no classes collapsed" in report_ok


def test_unknown_split_rejected(synthetic_data):
    with pytest.raises(ValueError):
        UltrasoundDataset(
            metadata_csv=synthetic_data["metadata_csv"],
            image_dir=synthetic_data["image_dir"],
            split="bogus",
        )
