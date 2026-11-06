"""Geometry helpers for keypoint-based pose rules."""

from __future__ import annotations

import numpy as np


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees (vertex at ``b``), in [0, 180]."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0


def ramp(x: float, lo: float, hi: float) -> float:
    """0 at ``lo``, 1 at ``hi``, linear in between (works for lo > hi too)."""
    if lo == hi:
        return float(x >= hi)
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def bell(x: float, center: float, tolerance: float) -> float:
    """1 at ``center``, falling linearly to 0 at ``center ± tolerance``."""
    return float(max(0.0, 1.0 - abs(x - center) / tolerance))
