"""Train a Track A supervised classifier from a YAML config.

Usage:
    python -m bd_usbench.scripts.train --config bd_usbench/configs/track_a_dinov2_linear.yaml \
        [--epochs 10] [--batch-size 16] [--export torchscript]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bd_usbench.train.export import export_model
from bd_usbench.train.trainer import train_track_a
from bd_usbench.utils import get_logger, load_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size")
    parser.add_argument("--export", choices=["torchscript", "onnx", "none"], default=None,
                        help="Export the best model after training (default: config export.format)")
    args = parser.parse_args(argv)

    overrides: dict = {"train": {}}
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["train"]["batch_size"] = args.batch_size
    cfg = load_config(args.config, overrides)

    summary = train_track_a(cfg)

    export_format = args.export or cfg.get("export", {}).get("format", "torchscript")
    if export_format != "none":
        run_dir = Path(cfg["output"]["dir"]) / cfg["run_name"]
        paths = export_model(summary["best_checkpoint"], run_dir / "export", export_format)
        summary["exported_model"] = str(paths["model"])

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
