"""Synthetic dataset generator: random images + fake metadata.csv.

Used by the pytest suite and by ``scripts/make_synthetic.py`` so the whole
pipeline can be exercised without any real (private) ultrasound data.
Each class gets a distinct geometric pattern so a small model can actually
learn to separate them in a couple of epochs.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from bd_usbench.utils import ensure_dir, get_logger

logger = get_logger(__name__)

DEFAULT_CLASSES = ("liver", "kidney", "bladder", "heart", "fetal_head")


def _draw_class_pattern(img: np.ndarray, class_idx: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a class-specific bright shape on a noisy grayscale image."""
    h, w = img.shape
    cx = int(w * (0.25 + 0.5 * rng.random()))
    cy = int(h * (0.25 + 0.5 * rng.random()))
    size = int(min(h, w) * (0.15 + 0.1 * rng.random()))
    shape = class_idx % 5
    if shape == 0:
        cv2.circle(img, (cx, cy), size, 230, -1)
    elif shape == 1:
        cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), 220, -1)
    elif shape == 2:
        cv2.ellipse(img, (cx, cy), (size, size // 2), 0, 0, 360, 210, -1)
    elif shape == 3:
        pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]])
        cv2.fillPoly(img, [pts], 240)
    else:
        cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), 250, max(2, size // 3))
        cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), 250, max(2, size // 3))
    return img


def make_synthetic_dataset(
    root: str | Path,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    images_per_patient: int = 3,
    train_patients_per_class: int = 2,
    val_patients_per_class: int = 1,
    test_patients_per_class: int = 1,
    image_size: int = 96,
    seed: int = 42,
) -> dict[str, Path]:
    """Create a synthetic image folder + metadata.csv under ``root``.

    Returns a dict with keys ``metadata_csv`` and ``image_dir``.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    image_dir = ensure_dir(root / "images")
    rows: list[dict] = []
    patient_counter = 0

    split_plan = (
        [("train", train_patients_per_class),
         ("val", val_patients_per_class),
         ("test", test_patients_per_class)]
    )
    for class_idx, organ in enumerate(classes):
        domain = "fetal" if organ.startswith("fetal") else "adult"
        for split, n_patients in split_plan:
            for _ in range(n_patients):
                patient_counter += 1
                patient_id = f"p{patient_counter:04d}"
                for img_idx in range(images_per_patient):
                    noise = rng.integers(0, 90, (image_size, image_size), dtype=np.uint8)
                    img = _draw_class_pattern(noise.copy(), class_idx, rng)
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                    filename = f"{patient_id}_{organ}_{img_idx}.png"
                    cv2.imwrite(str(image_dir / filename), img)
                    rows.append(
                        {
                            "filename": filename,
                            "patient_id": patient_id,
                            "primary_organ": organ,
                            "secondary_organ": "bowel;spine" if img_idx == 0 else "none",
                            "finding": "none",
                            "plane": "transverse",
                            "machine": "synthetic_machine",
                            "quality": "good",
                            "confidence": "high",
                            "labeled_by": "synthetic",
                            "reviewed_by": "none",
                            "split": split,
                            "domain": domain,
                        }
                    )

    metadata_csv = root / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata_csv, index=False)
    logger.info(
        "Synthetic dataset: %d images, %d classes, at %s", len(rows), len(classes), root
    )
    return {"metadata_csv": metadata_csv, "image_dir": image_dir}
