"""Dataset class for BD-USBench ultrasound images with metadata.csv labels."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from bd_usbench.utils import get_logger, load_json, save_json

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "filename",
    "patient_id",
    "primary_organ",
    "secondary_organ",
    "finding",
    "plane",
    "machine",
    "quality",
    "confidence",
    "labeled_by",
    "reviewed_by",
    "split",
    "domain",
]

VALID_SPLITS = ("train", "val", "test")
VALID_DOMAINS = ("fetal", "adult")

# Bucket label for rare classes collapsed by ``collapse_rare_classes``.
OTHER_CLASS = "other"


def load_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    """Load and normalize metadata.csv.

    Labels are lowercased/stripped; ``filename`` is kept verbatim (only
    stripped) because image files are opened by that exact name. The string
    ``none`` means empty for optional columns; ``keep_default_na=False`` so
    pandas never turns literal strings like "none"/"None"/"NA" into NaN.
    Raises ValueError on missing columns or bad values.
    """
    df = pd.read_csv(metadata_csv, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {missing}")
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
        if col != "filename":
            df[col] = df[col].str.lower()
        df[col] = df[col].replace("", "none")
    bad_splits = sorted(set(df["split"]) - set(VALID_SPLITS))
    if bad_splits:
        raise ValueError(f"Unknown split values: {bad_splits} (expected {VALID_SPLITS})")
    bad_domains = sorted(set(df["domain"]) - set(VALID_DOMAINS))
    if bad_domains:
        raise ValueError(f"Unknown domain values: {bad_domains} (expected {VALID_DOMAINS})")
    if (df["primary_organ"] == "none").any():
        n = int((df["primary_organ"] == "none").sum())
        logger.warning("Dropping %d rows with primary_organ == 'none'", n)
        df = df[df["primary_organ"] != "none"]
    return df.reset_index(drop=True)


def parse_secondary(value: str) -> list[str]:
    """Parse a secondary_organ cell: 'none' -> [], 'a;b' -> ['a', 'b']."""
    if value in ("", "none"):
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def collapse_rare_classes(
    df: pd.DataFrame, min_train_images: int
) -> tuple[pd.DataFrame, list[str]]:
    """Remap classes with fewer than ``min_train_images`` TRAIN images to
    :data:`OTHER_CLASS`, across all splits, so metrics are not dominated by
    classes that cannot be learned or validated.

    Counts are taken on the train split of the FULL dataframe, so the mapping
    is identical for train/val/test and for every track. Returns the (possibly
    copied) dataframe and the sorted list of collapsed class names.
    """
    if min_train_images <= 0:
        return df, []
    train_counts = df.loc[df["split"] == "train", "primary_organ"].value_counts()
    rare = sorted(
        c
        for c in df["primary_organ"].unique()
        if c != OTHER_CLASS and int(train_counts.get(c, 0)) < min_train_images
    )
    if not rare:
        return df, []
    df = df.copy()
    df.loc[df["primary_organ"].isin(rare), "primary_organ"] = OTHER_CLASS
    return df, rare


def build_class_list(df: pd.DataFrame, min_train_images: int = 0) -> list[str]:
    """Sorted, de-duplicated primary_organ labels across the whole CSV.

    With ``min_train_images > 0``, rare classes are collapsed into
    :data:`OTHER_CLASS` first (see :func:`collapse_rare_classes`).
    """
    df, _ = collapse_rare_classes(df, min_train_images)
    return sorted(df["primary_organ"].unique().tolist())


def save_classes(classes: list[str], path: str | Path) -> None:
    """Persist the class list to classes.json so all tracks share indices."""
    save_json({"classes": classes}, path)
    logger.info("Saved %d classes to %s", len(classes), path)


def load_classes(path: str | Path) -> list[str]:
    """Load the class list written by :func:`save_classes`."""
    return load_json(path)["classes"]


def load_image(path: str | Path, crop_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Read an image as grayscale, optionally crop, and return 3-channel RGB.

    ``crop_box`` is (x, y, width, height) in pixels, applied before resizing;
    use it to remove machine UI / burned-in text around the scan area.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if crop_box is not None:
        x, y, w, h = (int(v) for v in crop_box)
        img = img[y : y + h, x : x + w]
        if img.size == 0:
            raise ValueError(f"crop_box {crop_box} produced an empty image for {path}")
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


class UltrasoundDataset(Dataset):
    """B-mode ultrasound still-image dataset.

    Reads metadata.csv + an image folder, filters by split and optionally by
    domain, and maps ``primary_organ`` to class indices. The class list is
    always derived from (or given for) the FULL csv so that indices are
    identical across splits, domains, and tracks.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        image_dir: str | Path,
        split: str | None = None,
        domain: str = "all",
        classes: list[str] | None = None,
        transform: Callable | None = None,
        crop_box: tuple[int, int, int, int] | None = None,
        min_train_images: int = 0,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.crop_box = tuple(crop_box) if crop_box else None

        full_df = load_metadata(metadata_csv)
        full_df, self.collapsed_classes = collapse_rare_classes(full_df, min_train_images)
        self.classes = list(classes) if classes is not None else build_class_list(full_df)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        df = full_df
        if split is not None:
            if split not in VALID_SPLITS:
                raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
            df = df[df["split"] == split]
        if domain != "all":
            if domain not in VALID_DOMAINS:
                raise ValueError(f"domain must be 'fetal', 'adult' or 'all', got {domain!r}")
            df = df[df["domain"] == domain]

        unknown = sorted(set(df["primary_organ"]) - set(self.classes))
        if unknown:
            raise ValueError(f"Labels not in class list: {unknown}")

        self.df = df.reset_index(drop=True)
        logger.info(
            "Dataset: %d images (split=%s, domain=%s, %d classes)",
            len(self.df), split or "all", domain, len(self.classes),
        )

    def __len__(self) -> int:
        return len(self.df)

    def image_path(self, idx: int) -> Path:
        """Absolute path of the image at ``idx``."""
        return self.image_dir / self.df.iloc[idx]["filename"]

    def label_index(self, idx: int) -> int:
        """Class index of the sample at ``idx``."""
        return self.class_to_idx[self.df.iloc[idx]["primary_organ"]]

    def get_metadata(self, idx: int) -> dict:
        """Full metadata row (with parsed secondary organs) for a sample."""
        row = self.df.iloc[idx].to_dict()
        row["secondary_organs"] = parse_secondary(row["secondary_organ"])
        return row

    def class_counts(self) -> torch.Tensor:
        """Per-class sample counts in this (filtered) dataset."""
        counts = torch.zeros(len(self.classes), dtype=torch.long)
        for organ, n in self.df["primary_organ"].value_counts().items():
            counts[self.class_to_idx[organ]] = int(n)
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted cross-entropy."""
        counts = self.class_counts().float().clamp(min=1.0)
        weights = counts.sum() / (len(self.classes) * counts)
        return weights

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = load_image(self.image_path(idx), self.crop_box)
        pil = Image.fromarray(img)
        tensor = self.transform(pil) if self.transform else torch.from_numpy(
            np.asarray(pil).transpose(2, 0, 1)
        ).float().div_(255.0)
        return tensor, self.label_index(idx)
