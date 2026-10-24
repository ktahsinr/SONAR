"""Print the dataset report: per-class counts per split, low-count flags,
patient split-leakage check, and (optionally) missing-file check.

Usage:
    python -m bd_usbench.scripts.data_report --metadata data/metadata.csv \
        --image-dir data/images [--min-train 30] [--out report.txt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bd_usbench.data.report import build_report, patient_split_leakage
from bd_usbench.data.dataset import load_metadata
from bd_usbench.utils import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--image-dir", default=None, help="Image folder (enables missing-file check)")
    parser.add_argument("--min-train", type=int, default=30,
                        help="Flag classes with fewer train images than this (default 30)")
    parser.add_argument("--out", default=None, help="Also write the report to this file")
    args = parser.parse_args(argv)

    report = build_report(args.metadata, image_dir=args.image_dir, min_train=args.min_train)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        logger.info("Report written to %s", args.out)

    # Non-zero exit on split leakage so CI / notebooks fail loudly.
    leaks = patient_split_leakage(load_metadata(args.metadata))
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
