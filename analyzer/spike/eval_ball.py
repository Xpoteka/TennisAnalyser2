"""Score ball runs (ball_run.py output) against the CVAT ball labels.

Per labeled frame, with the ball within TOLERANCE of the label counting as found:
  ball labeled, predicted close      -> TP
  ball labeled, predicted elsewhere  -> FP and FN (a wrong position is a made-up ball too)
  ball labeled, nothing predicted    -> FN
  no ball labeled, ball predicted    -> FP, also counted as an "invented ball"
  no ball labeled, nothing predicted -> TN
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import calibration
from labels import BallLabels

# TrackNet's papers count a ball as found within 4 px at 1280x720; scaled with the frame width
# that is 12 px on 4K.
TOLERANCE_PX_AT_1280 = 4.0
MAX_INVENTED_RATE = 0.01  # pass mark: invented balls at the 0.95 threshold
MIN_COVERAGE = 0.60  # pass mark: coverage at 95% precision


def read_run(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    with path.open() as f:
        meta = json.loads(f.readline())["meta"]
        records = {}
        for line in f:
            if line.strip():
                r = json.loads(line)
                records[r["frame"]] = r
    return meta, records


def tolerance(width: int) -> float:
    return TOLERANCE_PX_AT_1280 * width / 1280


def combo_of(meta: dict[str, Any]) -> str:
    return f"{meta['model']}.{meta['runtime']}.{meta['input'][0]}.{meta['precision']}"


@dataclass
class FrameResult:
    clip: str
    group: str
    frame: int
    split: str
    gt: bool  # a ball is labeled on this frame
    predicted: bool
    correct: bool  # predicted and within tolerance
    score: float | None


def score_frames(clip: str, group: str, labels: BallLabels, records: dict[int, dict], width: int) -> tuple[list[FrameResult], int]:
    """Results for every labeled frame the run covered, and how many labeled frames it did not cover."""
    tol = tolerance(width)
    out, missing = [], 0
    for frame in labels.frames():
        if frame not in records:
            missing += 1
            continue
        ball = records[frame].get("ball")
        pred = ball is not None and ball.get("x") is not None and ball.get("y") is not None
        gt = labels.points.get(frame)
        correct = pred and gt is not None and math.dist((ball["x"], ball["y"]), gt) <= tol
        score = ball.get("score_raw") if pred else None
        out.append(FrameResult(clip, group, frame, calibration.split_of(frame), gt is not None, pred, bool(correct), score))
    return out, missing


def counts(results: Iterable[FrameResult]) -> dict[str, Any]:
    rs = list(results)
    tp = sum(r.correct for r in rs)
    fp = sum(r.predicted and not r.correct for r in rs)
    fn = sum(r.gt and not r.correct for r in rs)
    empty = [r for r in rs if not r.gt]
    invented = sum(r.predicted for r in empty)
    p = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * rec / (p + rec) if p and rec else None
    rnd = lambda v: None if v is None else round(v, 4)  # noqa: E731
    return {
        "frames": len(rs), "ball_visible": sum(r.gt for r in rs), "no_ball": len(empty),
        "tp": tp, "fp": fp, "fn": fn, "precision": rnd(p), "recall": rnd(rec), "f1": rnd(f1),
        "invented": invented, "invented_rate": rnd(invented / len(empty)) if empty else None,
    }


def evaluate_combo(results: list[FrameResult]) -> dict[str, Any]:
    """Fit the calibration on the calibration half, report everything else on the test half, per group."""
    fit = [r for r in results if r.split == "calibrate" and r.predicted]
    cal = calibration.fit_isotonic([r.score for r in fit], [r.correct for r in fit])
    test = [r for r in results if r.split == "test"]
    report: dict[str, Any] = {"calibration": cal.to_json() if cal else None, "groups": {}}
    if cal is None:
        report["note"] = "not enough calibration-half predictions with both right and wrong answers; nothing shown as fact"

    for group in sorted({r.group for r in test}) + ["all"]:
        rs = [r for r in test if group == "all" or r.group == group]
        preds = [r for r in rs if r.predicted]
        conf = [cal(r.score) if cal else 0.0 for r in preds]
        fact = calibration.at_threshold(conf, [r.correct for r in preds], sum(r.gt for r in rs))
        empty = [r for r in rs if not r.gt]
        invented = sum(1 for r, c in zip(preds, conf) if not r.gt and c >= calibration.FACT_THRESHOLD)
        fact["invented"] = invented
        fact["invented_rate"] = round(invented / len(empty), 4) if empty else None
        gates = {
            "coverage_at_95": fact["coverage"] is not None and fact["coverage"] >= MIN_COVERAGE
            and fact["precision"] is not None and fact["precision"] >= calibration.FACT_THRESHOLD,
            "invented_balls": fact["invented_rate"] is not None and fact["invented_rate"] <= MAX_INVENTED_RATE,
        }
        report["groups"][group] = {
            "raw": counts(rs),
            "fact": fact,
            "reliability": calibration.reliability(conf, [r.correct for r in preds]),
            "gates": gates,
            "pass": all(gates.values()),
        }
    return report
