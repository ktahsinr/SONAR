"""Export a trained Track A checkpoint to TorchScript or ONNX.

Usage:
    python -m bd_usbench.scripts.export_model \
        --checkpoint results/<run>/checkpoints/best.pt \
        [--out results/<run>/export] [--format torchscript|onnx]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bd_usbench.train.export import export_model
from bd_usbench.utils import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Track A training checkpoint (best.pt)")
    parser.add_argument("--out", default=None,
                        help="Output folder (default: <checkpoint dir>/../export)")
    parser.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript",
                        help="Export format (default torchscript)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent.parent / "export"
    paths = export_model(args.checkpoint, out_dir, args.format)
    print(f"Exported model: {paths['model']}\nMetadata: {paths['meta']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
