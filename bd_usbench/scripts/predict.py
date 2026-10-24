"""Predict organs on still images with an exported Track A model.

Prints predicted organ + confidence per image, flags "uncertain" below a
configurable threshold (default 0.6), and can save annotated copies.
This is a manual checking tool for still images — NOT a live/real-time tool.

Usage:
    python -m bd_usbench.scripts.predict \
        --model results/<run>/export/model.torchscript.pt \
        --input path/to/image_or_folder [--threshold 0.6] \
        [--save-annotated out_folder] [--json predictions.json]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from bd_usbench.predict.predictor import StillImagePredictor
from bd_usbench.utils import get_logger, save_json

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Exported model (model.torchscript.pt or model.onnx)")
    parser.add_argument("--input", required=True, help="Image file or folder of images")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Confidence below this is flagged uncertain (default 0.6)")
    parser.add_argument("--save-annotated", default=None,
                        help="Folder to save copies of the images with the prediction drawn on")
    parser.add_argument("--json", default=None, help="Also save predictions as JSON")
    args = parser.parse_args(argv)

    predictor = StillImagePredictor(args.model, threshold=args.threshold)
    predictions = predictor.predict_paths(args.input)

    for pred in predictions:
        flag = "  [UNCERTAIN]" if pred.uncertain else ""
        print(f"{pred.path.name}: {pred.organ} ({pred.confidence:.3f}){flag}")
        if args.save_annotated:
            out = StillImagePredictor.save_annotated(pred, args.save_annotated)
            logger.info("Annotated copy: %s", out)

    uncertain = sum(p.uncertain for p in predictions)
    print(f"\n{len(predictions)} image(s), {uncertain} flagged uncertain "
          f"(threshold {args.threshold}).")

    if args.json:
        save_json(
            [{**asdict(p), "path": str(p.path)} for p in predictions], Path(args.json)
        )
        logger.info("Predictions saved to %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
