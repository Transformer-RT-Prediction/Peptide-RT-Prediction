from __future__ import annotations

from typing import Tuple

import numpy as np


def pearson_r(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def p95_width(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.percentile(x, 97.5) - np.percentile(x, 2.5))


def residual_ci95(residuals) -> Tuple[float, float]:
    """95% confidence interval (2.5–97.5 percentile) of residuals in minutes."""
    r = np.asarray(residuals, dtype=np.float64)
    lo = float(np.percentile(r, 2.5))
    hi = float(np.percentile(r, 97.5))
    return lo, hi


def normalize_rt_units(y_raw, unit_mode: str, sec_to_min_threshold: float):
    """
    Convert RT units based on configuration and data spread.
    """
    y = np.asarray(y_raw, dtype=np.float64)
    dt95 = p95_width(y)

    if unit_mode == "minutes":
        return y, "minutes"
    if unit_mode == "seconds":
        return y / 60.0, "seconds→minutes(/60)"

    if dt95 > sec_to_min_threshold:
        return y / 60.0, "seconds→minutes(/60)"
    else:
        return y, "minutes"


__all__ = [
    "pearson_r",
    "p95_width",
    "residual_ci95",
    "normalize_rt_units",
]

