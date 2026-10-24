"""Mean per-image inference latency on the current device."""

from __future__ import annotations

import time
from typing import Callable


def measure_latency(
    predict_fn: Callable[[], object], n_warmup: int = 3, n_runs: int = 20
) -> dict:
    """Time ``predict_fn`` (a zero-arg single-image prediction) and return
    mean/std milliseconds per image."""
    for _ in range(n_warmup):
        predict_fn()
    times: list[float] = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn()
        times.append((time.perf_counter() - start) * 1000.0)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return {"mean_ms": round(mean, 3), "std_ms": round(std, 3), "n_runs": n_runs}
