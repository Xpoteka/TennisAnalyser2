"""Run every ball candidate on every clip in a samples folder, one after the other.

    python ball_matrix.py --samples /work/samples --start-frame 0 --max-frames 600
    python ball_matrix.py --samples /work/samples --labels /work/labels   # exactly the frames you labeled

With --labels, each clip that has a <clip>.ball.xml is run once per labeled segment, and clips
without ball labels are skipped.

A failing combination is logged in out/ball/matrix.json and the next one still runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import ball_run
import labels as L
from pose_matrix import find_videos

# (model, runtime, input width, precision)
COMBOS = [
    ("wasb", "xpu", 512, "fp32"),
    ("wasb", "openvino", 512, "fp32"),
    ("wasb", "openvino", 512, "fp16"),
    ("wasb", "openvino", 1024, "fp32"),
    ("tracknetv2", "xpu", 512, "fp32"),
    ("tracknetv2", "openvino", 512, "fp32"),
    ("yolo-ball", "openvino", 512, "fp32"),
]


def ranges(videos: list[Path], labels_dir: Path | None, start: int, count: int | None) -> list[tuple[Path, int, int | None]]:
    """(video, start frame, frame count) to run. From the labeled segments when a labels folder is given."""
    if labels_dir is None:
        return [(v, start, count) for v in videos]
    out = []
    for video in videos:
        path = labels_dir / f"{video.stem}.ball.xml"
        if path.is_file():
            out += [(video, a, b - a + 1) for a, b in L.load_ball_xml(path).segments]
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", type=Path, default=Path("samples"))
    p.add_argument("--out-dir", type=Path, default=Path("out/ball"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--labels", type=Path, default=None, help="run only the labeled ball segments")
    a = p.parse_args(argv)

    videos = find_videos(a.samples)
    if not videos:
        print(f"error: no videos in {a.samples}", file=sys.stderr)
        return 2
    todo = ranges(videos, a.labels, a.start_frame, a.max_frames)
    if not todo:
        print(f"error: no <clip>.ball.xml in {a.labels} matches a video in {a.samples}", file=sys.stderr)
        return 2

    results = []
    for video, start, count in todo:
        for model, runtime, width, precision in COMBOS:
            cfg = ball_run.BallConfig(video, model, runtime, width, precision, start, count,
                                      a.out_dir, a.models_dir)
            print(f"== {cfg.run_name}", flush=True)
            try:
                ball_run.run(cfg)
                timing = json.loads((a.out_dir / f"{cfg.run_name}.timing.json").read_text())
                results.append({"run": cfg.run_name, "ok": True, "e2e_fps": timing["e2e_fps"]})
            except Exception as exc:  # keep going: one broken model must not hide the others' results
                traceback.print_exc()
                results.append({"run": cfg.run_name, "ok": False, "error": f"{type(exc).__name__}: {exc}"})

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "matrix.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
