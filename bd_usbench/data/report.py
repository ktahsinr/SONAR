"""Dataset sanity report: per-class counts per split, low-count flags,
patient-level split leakage check, missing-file check."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bd_usbench.data.dataset import OTHER_CLASS, collapse_rare_classes, load_metadata
from bd_usbench.utils import get_logger

logger = get_logger(__name__)


def class_split_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table of counts: rows = primary_organ, columns = split."""
    table = (
        df.pivot_table(
            index="primary_organ", columns="split", values="filename",
            aggfunc="count", fill_value=0,
        )
        .reindex(columns=["train", "val", "test"], fill_value=0)
        .astype(int)
    )
    table["total"] = table.sum(axis=1)
    return table.sort_values("total", ascending=False)


def low_count_classes(df: pd.DataFrame, min_train: int = 30) -> list[tuple[str, int]]:
    """Classes with fewer than ``min_train`` training images."""
    counts = class_split_counts(df)
    low = counts[counts["train"] < min_train]
    return [(organ, int(n)) for organ, n in low["train"].items()]


def patient_split_leakage(df: pd.DataFrame) -> list[str]:
    """Patient IDs that appear in more than one split (should be empty)."""
    splits_per_patient = df.groupby("patient_id")["split"].nunique()
    return sorted(splits_per_patient[splits_per_patient > 1].index.tolist())


def missing_files(df: pd.DataFrame, image_dir: str | Path) -> list[str]:
    """Filenames referenced in the CSV that do not exist in ``image_dir``."""
    image_dir = Path(image_dir)
    return [f for f in df["filename"] if not (image_dir / f).exists()]


def build_report(
    metadata_csv: str | Path,
    image_dir: str | Path | None = None,
    min_train: int = 30,
    min_train_images: int = 5,
) -> str:
    """Build the full human-readable data report as a string."""
    df = load_metadata(metadata_csv)
    lines: list[str] = []
    lines.append(f"BD-USBench data report — {metadata_csv}")
    lines.append(f"Images: {len(df)} | Patients: {df['patient_id'].nunique()} "
                 f"| Classes: {df['primary_organ'].nunique()}")
    lines.append("")

    lines.append("Per-class counts per split:")
    lines.append(class_split_counts(df).to_string())
    lines.append("")

    domain_table = df.pivot_table(
        index="domain", columns="split", values="filename", aggfunc="count", fill_value=0
    ).astype(int)
    lines.append("Per-domain counts per split:")
    lines.append(domain_table.to_string())
    lines.append("")

    _, collapsed = collapse_rare_classes(df, min_train_images)
    if collapsed:
        train_counts = df.loc[df["split"] == "train", "primary_organ"].value_counts()
        lines.append(
            f"Classes collapsed into '{OTHER_CLASS}' "
            f"(< {min_train_images} train images): {len(collapsed)}"
        )
        for organ in collapsed:
            lines.append(f"  - {organ}: {int(train_counts.get(organ, 0))} train images")
    elif min_train_images > 0:
        lines.append(
            f"OK: no classes collapsed (all have >= {min_train_images} train images)."
        )
    lines.append("")

    low = low_count_classes(df, min_train=min_train)
    if low:
        lines.append(f"WARNING: classes with < {min_train} train images:")
        for organ, n in low:
            lines.append(f"  - {organ}: {n} train images")
    else:
        lines.append(f"OK: all classes have >= {min_train} train images.")
    lines.append("")

    leaks = patient_split_leakage(df)
    if leaks:
        lines.append(f"ERROR: {len(leaks)} patient_id(s) cross splits: {leaks}")
    else:
        lines.append("OK: no patient_id crosses splits.")

    if image_dir is not None:
        miss = missing_files(df, image_dir)
        if miss:
            lines.append(f"ERROR: {len(miss)} file(s) in CSV missing on disk "
                         f"(first 10): {miss[:10]}")
        else:
            lines.append("OK: all files referenced in CSV exist on disk.")

    return "\n".join(lines)
