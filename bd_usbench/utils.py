"""Shared utilities: logging, seeding, config loading, device resolution."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring root logging once."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
        root.setLevel(os.environ.get("BDUS_LOG_LEVEL", "INFO"))
    return logging.getLogger(name)


def set_seed(seed: int = 42) -> None:
    """Fix all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve a device string ('auto', 'cpu', 'cuda', 'mps') to a torch.device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load a YAML config file, optionally merging in overrides."""
    with open(path, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as pretty-printed JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
