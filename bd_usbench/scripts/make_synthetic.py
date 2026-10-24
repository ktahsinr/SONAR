"""Generate a synthetic dataset (random images + fake metadata.csv) for local
smoke runs without any real data.

Usage:
    python -m bd_usbench.scripts.make_synthetic --out data_synthetic [--seed 42]
"""

from __future__ import annotations

import argparse
import sys

from bd_usbench.data.synthetic import make_synthetic_dataset
from bd_usbench.utils import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output folder")
    parser.add_argument("--images-per-patient", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    paths = make_synthetic_dataset(
        args.out,
        images_per_patient=args.images_per_patient,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(f"metadata: {paths['metadata_csv']}\nimages:   {paths['image_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
