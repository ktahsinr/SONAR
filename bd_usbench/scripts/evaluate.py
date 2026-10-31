"""Evaluate any track on a chosen split/domain; writes results/<run_name>/eval_*/.

Usage:
    python -m bd_usbench.scripts.evaluate --track a \
        --config bd_usbench/configs/track_a_dinov2_linear.yaml \
        [--split test] [--domain fetal|adult|all] [--checkpoint path/to/best.pt] \
        [--results-dir results]

Track A loads eval.checkpoint if set, else results/<run_name>/checkpoints/best.pt;
Track C loads the LoRA adapter from results/<run_name>/adapter. --results-dir
points output.dir at a different root, so evaluation can run against results/
copied in from a previous session without retraining.

Note: evaluate on the test split only after val-based model selection.
"""

from __future__ import annotations

import argparse
import json
import sys

from bd_usbench.eval.runners import evaluate
from bd_usbench.utils import get_logger, load_config

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=["a", "b", "c"], help="Track to evaluate")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Split to evaluate (default: test)")
    parser.add_argument("--domain", default="all", choices=["fetal", "adult", "all"],
                        help="Domain filter (default: all)")
    parser.add_argument("--checkpoint", default=None,
                        help="Track A: explicit checkpoint path (default: run's best.pt)")
    parser.add_argument("--results-dir", default=None,
                        help="Override output.dir: root folder holding all runs' results")
    args = parser.parse_args(argv)

    overrides: dict = {}
    if args.checkpoint:
        overrides["eval"] = {"checkpoint": args.checkpoint}
    if args.results_dir:
        overrides["output"] = {"dir": args.results_dir}
    cfg = load_config(args.config, overrides)

    metrics = evaluate(args.track, cfg, split=args.split, domain=args.domain)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
