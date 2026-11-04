"""Run every pose candidate on every clip in a samples folder, one after the other.

    python pose_matrix.py --samples /work/samples --max-frames 600

A failing combination is logged in out/pose/matrix.json and the next one still runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pose_run

# (model, size, runtime): the spike plan's pose candidates on both runtimes where they exist.
COMBOS = [
    ("yolo11-pose", "m", "xpu"),
    ("yolo11-pose", "m", "openvino"),
    ("yolo11-pose", "x", "xpu"),
    ("yolo11-pose", "x", "openvino"),
    ("rtmpose", "m", "openvino"),
    ("rtmpose", "x", "openvino"),
]
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}


def find_videos(samples: Path) -> list[Path]:
    return sorted(p for p in samples.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--samples", type=Path, default=Path("samples"))
    p.add_argument("--out-dir", type=Path, default=Path("out/pose"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=None)
    a = p.parse_args(argv)

    videos = find_videos(a.samples)
    if not videos:
        print(f"error: no videos in {a.samples}", file=sys.stderr)
        return 2

    results = []
    for video in videos:
        for model, size, runtime in COMBOS:
            cfg = pose_run.RunConfig(video, model, size, runtime, a.imgsz, a.stride, a.max_frames,
                                     a.out_dir, a.models_dir)
            print(f"== {cfg.run_name}", flush=True)
            try:
                pose_run.run(cfg)
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
