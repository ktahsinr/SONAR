"""Single-image / batch prediction on still images with an exported model.

Loads a TorchScript (or ONNX) export plus its export_meta.json, replicates the
eval preprocessing, and optionally writes annotated copies of the images.
This is a manual checking tool for still images only — real-time / live
inference is explicitly out of scope for this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from bd_usbench.data.dataset import load_image
from bd_usbench.utils import get_logger, load_json

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class Prediction:
    """One still-image prediction."""

    path: Path
    organ: str
    confidence: float
    uncertain: bool
    probs: dict[str, float]


class StillImagePredictor:
    """CPU-friendly predictor for exported Track A models."""

    def __init__(self, model_path: str | Path, threshold: float = 0.6) -> None:
        model_path = Path(model_path)
        meta_path = model_path.parent / "export_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"export_meta.json not found next to {model_path} — "
                "export the model with scripts/export_model.py first."
            )
        self.meta = load_json(meta_path)
        self.classes: list[str] = self.meta["classes"]
        self.image_size: int = int(self.meta["image_size"])
        self.crop_box = self.meta.get("crop_box")
        self.mean = np.array(self.meta["normalize_mean"], dtype=np.float32)
        self.std = np.array(self.meta["normalize_std"], dtype=np.float32)
        self.threshold = float(threshold)

        if self.meta["format"] == "torchscript":
            self.model = torch.jit.load(str(model_path), map_location="cpu")
            self.model.eval()
            self._runtime = "torchscript"
        elif self.meta["format"] == "onnx":
            import onnxruntime  # lazy optional dependency

            self.session = onnxruntime.InferenceSession(str(model_path))
            self._runtime = "onnx"
        else:
            raise ValueError(f"Unknown export format {self.meta['format']!r}")
        logger.info(
            "Predictor loaded (%s, %d classes, input %dpx)",
            self._runtime, len(self.classes), self.image_size,
        )

    def _preprocess(self, path: str | Path) -> np.ndarray:
        rgb = load_image(path, self.crop_box)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        return arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)

    def predict(self, path: str | Path) -> Prediction:
        """Predict the primary organ of one still image."""
        batch = self._preprocess(path)
        if self._runtime == "torchscript":
            with torch.no_grad():
                logits = self.model(torch.from_numpy(batch)).numpy()
        else:
            input_name = self.session.get_inputs()[0].name
            logits = self.session.run(None, {input_name: batch})[0]
        exp = np.exp(logits - logits.max())
        probs = (exp / exp.sum())[0]
        idx = int(probs.argmax())
        confidence = float(probs[idx])
        return Prediction(
            path=Path(path),
            organ=self.classes[idx],
            confidence=confidence,
            uncertain=confidence < self.threshold,
            probs={c: float(p) for c, p in zip(self.classes, probs)},
        )

    def predict_paths(self, input_path: str | Path) -> list[Prediction]:
        """Predict one image file, or every image in a folder (sorted)."""
        input_path = Path(input_path)
        if input_path.is_dir():
            files = sorted(
                p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not files:
                raise FileNotFoundError(f"No images found in folder {input_path}")
        else:
            files = [input_path]
        return [self.predict(p) for p in files]

    @staticmethod
    def save_annotated(prediction: Prediction, out_dir: str | Path) -> Path:
        """Save a copy of the image with the prediction written on it."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(prediction.path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {prediction.path}")
        label = f"{prediction.organ} {prediction.confidence:.2f}"
        if prediction.uncertain:
            label += " (UNCERTAIN)"
        color = (0, 200, 255) if prediction.uncertain else (0, 220, 0)
        scale = max(0.5, img.shape[1] / 800.0)
        cv2.putText(img, label, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.8, (0, 0, 0), int(3 * scale))
        cv2.putText(img, label, (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.8, color, max(1, int(scale)))
        out_path = out_dir / f"{prediction.path.stem}_pred{prediction.path.suffix}"
        cv2.imwrite(str(out_path), img)
        return out_path
