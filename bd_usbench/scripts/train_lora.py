"""Track C: LoRA fine-tune a Qwen-VL model from a YAML config.

Usage:
    python -m bd_usbench.scripts.train_lora --config bd_usbench/configs/track_c_qwen_lora.yaml \
        [--epochs 3]
"""

from __future__ import annotations

import argparse
import json
import sys

from bd_usbench.train.lora import train_track_c
from bd_usbench.utils import get_logger, load_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    args = parser.parse_args(argv)

    overrides: dict = {"train": {}}
    if args.epochs is not None:
        overrides["train"]["epochs"] = args.epochs
    cfg = load_config(args.config, overrides)

    summary = train_track_c(cfg)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
