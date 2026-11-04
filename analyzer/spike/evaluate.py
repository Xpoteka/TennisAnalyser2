"""Phase 0, step 6: accuracy, calibrated confidence and runtime parity for every pose and ball run.

    python evaluate.py --runs /work/out --labels /work/labels --out /work/out/eval

Reads <runs>/ball/*.jsonl and <runs>/pose/*.jsonl, and the CVAT exports in <labels>
(<clip>.ball.xml, <clip>.pose.json, optional clips.json with each clip's camera height).
Writes report.json (everything), REPORT.md (the tables) and one calibration file per model
combination, which later becomes part of the stage.

Calibration is fitted on the calibration half and every reported number comes from the test half.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import eval_ball
import eval_pose
import labels as L
from eval_ball import read_run

PARITY_MIN_SCORE = 0.5  # parity is measured on keypoints both runtimes are reasonably sure of
MAX_PARITY_PX = 2.0  # pass mark: mean keypoint difference between XPU and OpenVINO


def clip_of(meta: dict[str, Any]) -> str:
    return Path(meta["video"]).stem


def pose_combo(meta: dict[str, Any]) -> str:
    return f"{meta['model']}-{meta['size']}.{meta['runtime']}.{meta['imgsz']}"


# ---------- parity: the same model on both runtimes must give the same answer ----------

def pose_parity(a: dict[int, dict], b: dict[int, dict]) -> dict[str, Any]:
    diffs = []
    for frame in sorted(a.keys() & b.keys()):
        for pa in a[frame]["players"]:
            best = None
            for pb in b[frame]["players"]:
                if None in pa["box"] or None in pb["box"]:
                    continue
                iou = _iou(pa["box"], pb["box"])
                if iou >= 0.5 and (best is None or iou > best[0]):
                    best = (iou, pb)
            if best is None:
                continue
            for ka, kb in zip(pa["keypoints"], best[1]["keypoints"]):
                if None in ka or None in kb or min(ka[2], kb[2]) < PARITY_MIN_SCORE:
                    continue
                diffs.append(math.dist(ka[:2], kb[:2]))
    out = {"keypoints": len(diffs), "mean_px": round(mean(diffs), 2) if diffs else None,
           "median_px": round(median(diffs), 2) if diffs else None, "min_score": PARITY_MIN_SCORE}
    out["pass"] = out["mean_px"] is not None and out["mean_px"] <= MAX_PARITY_PX
    return out


def ball_parity(a: dict[int, dict], b: dict[int, dict]) -> dict[str, Any]:
    frames = sorted(a.keys() & b.keys())
    found = lambda r: r.get("ball") is not None and r["ball"].get("x") is not None  # noqa: E731
    agree = sum(found(a[f]) == found(b[f]) for f in frames)
    dists = [math.dist((a[f]["ball"]["x"], a[f]["ball"]["y"]), (b[f]["ball"]["x"], b[f]["ball"]["y"]))
             for f in frames if found(a[f]) and found(b[f])]
    return {"frames": len(frames), "found_agree": round(agree / len(frames), 4) if frames else None,
            "both_found": len(dists), "mean_px": round(mean(dists), 2) if dists else None,
            "median_px": round(median(dists), 2) if dists else None}


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parity_pairs(runs: list[tuple[dict, dict]], key) -> list[tuple[str, dict, dict, dict, dict]]:
    """(name, meta_xpu, records_xpu, meta_ov, records_ov) for each model run on both runtimes on the same clip."""
    by = {}
    for meta, recs in runs:
        by[(clip_of(meta), key(meta), meta["runtime"])] = (meta, recs)
    out = []
    for (clip, k, rt), (meta, recs) in sorted(by.items(), key=lambda kv: kv[0]):
        if rt == "xpu" and (clip, k, "openvino") in by:
            ov_meta, ov_recs = by[(clip, k, "openvino")]
            out.append((f"{clip}.{k}", meta, recs, ov_meta, ov_recs))
    return out


# ---------- driver ----------

def load_runs(folder: Path) -> list[tuple[dict, dict]]:
    return [read_run(p) for p in sorted(folder.glob("*.jsonl"))] if folder.is_dir() else []


def evaluate_ball(runs, labels_dir: Path, groups: dict[str, str], segments: dict[str, list]) -> dict[str, Any]:
    by_combo: dict[str, list] = {}
    covered: dict[str, dict[str, set[int]]] = {}
    labeled: dict[str, set[int]] = {}
    info: dict[str, Any] = {"runs_without_labels": [], "labels": {}}
    for meta, recs in runs:
        clip = clip_of(meta)
        path = labels_dir / f"{clip}.ball.xml"
        if not path.is_file():
            info["runs_without_labels"].append(meta["run"])
            continue
        lab = L.load_ball_xml(path, segments=segments.get(clip))
        labeled[clip] = set(lab.frames())
        info["labels"][clip] = {"segments": lab.segments, "points": len(lab.points), "warnings": lab.warnings}
        combo = eval_ball.combo_of(meta)
        results, _ = eval_ball.score_frames(clip, groups.get(clip, "unknown"), lab, recs, meta["video_info"]["width"])
        done = covered.setdefault(combo, {}).setdefault(clip, set())
        fresh = [r for r in results if r.frame not in done]  # overlapping runs count each frame once
        done.update(r.frame for r in fresh)
        by_combo.setdefault(combo, []).extend(fresh)
    info["combos"] = {c: eval_ball.evaluate_combo(rs) for c, rs in sorted(by_combo.items())}
    # labeled frames no run of that combination covered: rerun ball_matrix.py --labels to cover them
    info["frames_not_run"] = {c: {clip: len(labeled[clip] - done) for clip, done in clips.items()}
                              for c, clips in covered.items()}
    key = lambda m: f"{m['model']}.{m['input'][0]}.{m['precision']}"  # noqa: E731
    info["parity"] = {name: ball_parity(ra, rb) for name, _, ra, _, rb in parity_pairs(runs, key)}
    return info


def evaluate_pose(runs, labels_dir: Path, groups: dict[str, str]) -> dict[str, Any]:
    by_combo: dict[str, list] = {}
    info: dict[str, Any] = {"runs_without_labels": [], "frames_not_run": {}}
    for meta, recs in runs:
        clip = clip_of(meta)
        path = labels_dir / f"{clip}.pose.json"
        if not path.is_file():
            info["runs_without_labels"].append(meta["run"])
            continue
        combo = pose_combo(meta)
        results, missing = eval_pose.score_frames(clip, groups.get(clip, "unknown"), L.load_pose_coco(path), recs,
                                                  meta["video_info"]["height"])
        by_combo.setdefault(combo, []).extend(results)
        info["frames_not_run"].setdefault(combo, {})[clip] = missing
    info["combos"] = {c: eval_pose.evaluate_combo(rs) for c, rs in sorted(by_combo.items())}
    key = lambda m: f"{m['model']}-{m['size']}.{m['imgsz']}"  # noqa: E731
    info["parity"] = {name: pose_parity(ra, rb) for name, _, ra, _, rb in parity_pairs(runs, key)}
    return info


def fmt(v, pct=False) -> str:
    if v is None:
        return "–"
    return f"{v * 100:.1f}%" if pct else str(v)


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Spike evaluation (test half)", "",
             "Coverage and precision are at calibrated confidence ≥ 0.95. \"lower\" is the 95% lower bound of that "
             "precision: with few samples the true precision can be that low.", "", "## Ball", "",
             "| Model | Height | Visible | F1 (raw) | Coverage | Precision (lower) | Invented | ECE | Pass |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for combo, r in report["ball"]["combos"].items():
        for group, g in r["groups"].items():
            f = g["fact"]
            lines.append(f"| {combo} | {group} | {g['raw']['ball_visible']} | {fmt(g['raw']['f1'], True)} | "
                         f"{fmt(f['coverage'], True)} | {fmt(f['precision'], True)} ({fmt(f['precision_lower_95'], True)}) | "
                         f"{fmt(f['invented_rate'], True)} | {fmt(g['reliability']['ece'])} | {'yes' if g['pass'] else 'no'} |")
    lines += ["", "## Pose (wrists and elbows)", "",
              "| Model | Height | Near PCK | Near coverage | Near precision (lower) | Far PCK | Far coverage | Pass |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for combo, r in report["pose"]["combos"].items():
        for group, g in r["groups"].items():
            n, fa = g["near"], g["far"]
            lines.append(f"| {combo} | {group} | {fmt(n['pck_arms'], True)} | {fmt(n['arms_fact']['coverage'], True)} | "
                         f"{fmt(n['arms_fact']['precision'], True)} ({fmt(n['arms_fact']['precision_lower_95'], True)}) | "
                         f"{fmt(fa['pck_arms'], True)} | {fmt(fa['arms_fact']['coverage'], True)} | {'yes' if g['pass'] else 'no'} |")
    lines += ["", "## Runtime parity (XPU vs OpenVINO)", "", "| Run | Mean px | Median px | Pass |", "| --- | --- | --- | --- |"]
    for name, p in report["pose"]["parity"].items():
        lines.append(f"| pose {name} | {fmt(p['mean_px'])} | {fmt(p['median_px'])} | {'yes' if p['pass'] else 'no'} |")
    for name, p in report["ball"]["parity"].items():
        lines.append(f"| ball {name} | {fmt(p['mean_px'])} | {fmt(p['median_px'])} | found agree {fmt(p['found_agree'], True)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", type=Path, default=Path("out"))
    p.add_argument("--labels", type=Path, default=Path("labels"))
    p.add_argument("--out", type=Path, default=Path("out/eval"))
    p.add_argument("--segments", action="append", default=[], metavar="CLIP=START-END[,START-END]",
                   help="ball frame ranges you labeled, if the automatic split by gaps is wrong")
    a = p.parse_args(argv)

    if not a.labels.is_dir():
        print(f"error: labels folder {a.labels} not found", file=sys.stderr)
        return 2
    try:
        segments = {}
        for s in a.segments:
            clip, _, ranges = s.partition("=")
            segments[clip] = L.parse_segments(ranges)
        groups = L.load_clip_groups(a.labels)
        report = {
            "ball": evaluate_ball(load_runs(a.runs / "ball"), a.labels, groups, segments),
            "pose": evaluate_pose(load_runs(a.runs / "pose"), a.labels, groups),
        }
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "report.json").write_text(json.dumps(report, indent=2))
    (a.out / "REPORT.md").write_text(markdown(report))
    for kind in ("ball", "pose"):
        for combo, r in report[kind]["combos"].items():
            if r["calibration"]:
                (a.out / f"calibration.{kind}.{combo}.json").write_text(json.dumps(r["calibration"], indent=2))
    print(markdown(report))
    print(f"Report written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
