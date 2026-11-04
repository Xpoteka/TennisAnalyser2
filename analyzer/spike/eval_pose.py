"""Score pose runs (pose_run.py output) against the CVAT COCO keypoint labels.

A keypoint is correct when it lies within PCK_ALPHA x the player's size of the label (PCK).
Only keypoints you labeled (visible or occluded) count; "outside" keypoints are left out.
Each labeled player is matched to the closest predicted person; an unmatched player has all
labeled keypoints missed.

Near or far player: the cameras stand behind the baselines, so the near player is in the lower
half of the frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import calibration
from labels import PosePlayer

KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
             "right_knee", "left_ankle", "right_ankle"]
ARMS = {"left_elbow", "right_elbow", "left_wrist", "right_wrist"}

# 5% of the player's size: about 9 cm on a 1.8 m player, roughly a wrist's width.
PCK_ALPHA = 0.05
# A labeled player whose mean keypoint error exceeds this share of their size is not the same person.
MAX_MATCH_DIST = 0.5
MIN_ARM_PCK = 0.90  # pass mark, near player
MIN_ARM_COVERAGE = 0.80  # pass mark, near player


@dataclass
class KeypointResult:
    clip: str
    group: str
    frame: int
    split: str
    player: str  # near | far
    keypoint: str
    visibility: int  # 1 occluded, 2 visible
    predicted: bool
    correct: bool
    score: float | None


def _pred_points(person: dict) -> list[tuple[float, float, float] | None]:
    out = []
    for kp in person["keypoints"]:
        out.append(None if any(v is None for v in kp) else (kp[0], kp[1], kp[2]))
    return out


def _match_cost(gt: PosePlayer, pred: list) -> float:
    dists = [math.dist((x, y), p[:2]) for (x, y, v), p in zip(gt.keypoints, pred) if v > 0 and p is not None]
    return sum(dists) / len(dists) / max(gt.size, 1.0) if dists else math.inf


def match_players(gts: list[PosePlayer], people: list[dict]) -> list[list | None]:
    """Greedy one-to-one match by mean normalised keypoint distance. With 2-4 people it equals the optimum
    in practice, and it needs no extra dependency."""
    preds = [_pred_points(p) for p in people]
    pairs = sorted((_match_cost(g, p), gi, pi) for gi, g in enumerate(gts) for pi, p in enumerate(preds))
    matched: list[list | None] = [None] * len(gts)
    used: set[int] = set()
    for cost, gi, pi in pairs:
        if cost > MAX_MATCH_DIST or matched[gi] is not None or pi in used:
            continue
        matched[gi] = preds[pi]
        used.add(pi)
    return matched


def side(player: PosePlayer, frame_height: int) -> str:
    centre_y = player.bbox[1] + player.bbox[3] / 2
    return "near" if centre_y > frame_height / 2 else "far"


def score_frames(clip: str, group: str, labels: dict[int, list[PosePlayer]], records: dict[int, dict],
                 frame_height: int) -> tuple[list[KeypointResult], int]:
    out, missing = [], 0
    for frame, gts in sorted(labels.items()):
        if frame not in records:
            missing += 1
            continue
        matched = match_players(gts, records[frame]["players"])
        for gt, pred in zip(gts, matched):
            who = side(gt, frame_height)
            for name, (x, y, v), p in zip(KEYPOINTS, gt.keypoints, pred or [None] * 17):
                if v == 0:
                    continue
                ok = p is not None and math.dist((x, y), p[:2]) <= PCK_ALPHA * gt.size
                out.append(KeypointResult(clip, group, frame, calibration.split_of(frame), who, name, v,
                                          p is not None, bool(ok), p[2] if p is not None else None))
    return out, missing


def pck(results: list[KeypointResult]) -> float | None:
    return round(sum(r.correct for r in results) / len(results), 4) if results else None


def evaluate_combo(results: list[KeypointResult]) -> dict[str, Any]:
    fit = [r for r in results if r.split == "calibrate" and r.predicted]
    cal = calibration.fit_isotonic([r.score for r in fit], [r.correct for r in fit])
    test = [r for r in results if r.split == "test"]
    report: dict[str, Any] = {"calibration": cal.to_json() if cal else None, "pck_alpha": PCK_ALPHA, "groups": {}}
    if cal is None:
        report["note"] = "not enough calibration-half keypoints with both right and wrong answers; nothing shown as fact"

    for group in sorted({r.group for r in test}) + ["all"]:
        rs = [r for r in test if group == "all" or r.group == group]
        g: dict[str, Any] = {}
        for who in ("near", "far"):
            mine = [r for r in rs if r.player == who]
            arms = [r for r in mine if r.keypoint in ARMS]
            arm_preds = [r for r in arms if r.predicted]
            conf = [cal(r.score) if cal else 0.0 for r in arm_preds]
            all_preds = [r for r in mine if r.predicted]
            all_conf = [cal(r.score) if cal else 0.0 for r in all_preds]
            g[who] = {
                "keypoints": len(mine),
                "pck": pck(mine),
                "pck_visible": pck([r for r in mine if r.visibility == 2]),
                "pck_occluded": pck([r for r in mine if r.visibility == 1]),
                "pck_arms": pck(arms),
                "per_keypoint": {k: pck([r for r in mine if r.keypoint == k]) for k in KEYPOINTS},
                "arms_fact": calibration.at_threshold(conf, [r.correct for r in arm_preds], len(arms)),
                "reliability": calibration.reliability(all_conf, [r.correct for r in all_preds]),
            }
        near = g["near"]
        fact = near["arms_fact"]
        g["gates"] = {
            "near_arm_pck": near["pck_arms"] is not None and near["pck_arms"] >= MIN_ARM_PCK,
            "near_arm_coverage_at_95": fact["coverage"] is not None and fact["coverage"] >= MIN_ARM_COVERAGE
            and fact["precision"] is not None and fact["precision"] >= calibration.FACT_THRESHOLD,
        }
        g["pass"] = all(g["gates"].values())
        report["groups"][group] = g
    return report
