"""Turn raw model scores into confidences that mean what they say.

1. Split the labeled frames into a calibration half and a test half. Frames are split in blocks of
   one second, alternating, so neighbouring (almost identical) frames never sit on both sides.
2. On the calibration half, fit isotonic regression: raw score -> share correct. It only assumes
   that a higher score is never worse, nothing about the shape of the curve.
3. On the test half, check the result with a reliability table and the operational number:
   coverage at the 0.95 threshold, and the precision actually reached there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

SPLIT_BLOCK_FRAMES = 60
FACT_THRESHOLD = 0.95
RELIABILITY_BINS = 10


def split_of(frame: int, block: int = SPLIT_BLOCK_FRAMES) -> str:
    return "calibrate" if (frame // block) % 2 == 0 else "test"


@dataclass
class Calibrator:
    """Isotonic mapping from raw score to probability of being correct."""

    xs: list[float]
    ys: list[float]
    n: int

    def __call__(self, score: float) -> float:
        import numpy as np

        return float(np.interp(score, self.xs, self.ys))

    def to_json(self) -> dict:
        return {"method": "isotonic", "n_fit": self.n, "x": self.xs, "y": self.ys}


def fit_isotonic(scores: Sequence[float], correct: Sequence[bool]) -> Calibrator | None:
    """None when there is nothing to fit: no samples, or all right / all wrong gives no curve."""
    from sklearn.isotonic import IsotonicRegression

    if len(scores) < 2 or len(set(bool(c) for c in correct)) < 2:
        return None
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(scores, [float(c) for c in correct])
    return Calibrator([float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_], len(scores))


def reliability(confidences: Sequence[float], correct: Sequence[bool], bins: int = RELIABILITY_BINS) -> dict:
    """Per bin: mean confidence vs actual share correct. ECE = their weighted gap."""
    rows = []
    total = len(confidences)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        conf = sum(confidences[i] for i in idx) / len(idx)
        acc = sum(bool(correct[i]) for i in idx) / len(idx)
        ece += len(idx) / total * abs(conf - acc)
        rows.append({"bin": [lo, hi], "n": len(idx), "mean_confidence": round(conf, 4), "share_correct": round(acc, 4)})
    return {"bins": rows, "ece": round(ece, 4) if total else None, "n": total}


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float | None:
    """Lower end of the 95% interval for a proportion: how low the precision could really be."""
    if n == 0:
        return None
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((centre - margin) / (1 + z * z / n), 4)


def at_threshold(confidences: Sequence[float], correct: Sequence[bool], n_targets: int,
                 threshold: float = FACT_THRESHOLD) -> dict:
    """Predictions at or above the threshold are shown as fact.

    coverage = correct facts / everything that should have been found (n_targets),
    precision = correct facts / facts shown.
    """
    shown = [bool(c) for conf, c in zip(confidences, correct) if conf >= threshold]
    hits = sum(shown)
    return {
        "threshold": threshold,
        "shown": len(shown),
        "correct": hits,
        "targets": n_targets,
        "coverage": round(hits / n_targets, 4) if n_targets else None,
        "precision": round(hits / len(shown), 4) if shown else None,
        "precision_lower_95": wilson_lower(hits, len(shown)),
    }
