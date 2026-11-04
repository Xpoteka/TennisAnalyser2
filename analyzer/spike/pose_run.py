"""Phase 0, step 3: 2D pose on a sample clip, one model x runtime per run.

    python pose_run.py --video samples/rally.mp4 --model yolo11-pose --size x --runtime openvino

Models:
  yolo11-pose  one-stage: people + 17 COCO keypoints in one pass (Ultralytics)
  rtmpose      top-down: YOLO11 finds people, RTMPose (body7, ONNX via rtmlib) puts keypoints on each crop

Runtimes: `xpu` (PyTorch on the Arc GPU) or `openvino` (the discrete GPU, never the iGPU).
RTMPose ships as ONNX only, so it runs on OpenVINO; its YOLO11 person detector uses the same runtime.

Output, under --out-dir:
  <run>.jsonl         line 1 = run metadata, then one line per processed frame
  <run>.timing.json   per-stage mean ms per frame and end-to-end FPS

Scores are the models' raw outputs (`score_raw`), not calibrated confidences: that mapping is
fitted in step 6. A frame with no detected person has `"players": []`; nothing is interpolated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

MODELS = ("yolo11-pose", "rtmpose")
RUNTIMES = ("xpu", "openvino")
SIZES = ("m", "l", "x")

# Detections below this raw score are dropped. Deliberately low: the spike keeps uncertain
# people and lets the calibration step decide what can be shown.
MIN_BOX_SCORE = 0.25
PERSON_CLASS = 0

RTMPOSE_BODY7 = {
    # size: (ONNX package URL, model input (w, h))
    "m": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
          "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip", (192, 256)),
    "x": ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
          "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip", (288, 384)),
}


@dataclass
class Timing:
    """Per-stage time. The first frame (graph compilation, memory allocation) is reported separately."""

    frames: int = 0
    decode_s: float = 0.0
    infer_s: float = 0.0
    wall_s: float = 0.0
    first_frame_s: float | None = None

    def summary(self) -> dict[str, Any]:
        steady = self.frames - 1
        per_frame = lambda s: round(1000 * s / steady, 2) if steady > 0 else None  # noqa: E731
        return {
            "frames": self.frames,
            "first_frame_ms": round(1000 * self.first_frame_s, 1) if self.first_frame_s is not None else None,
            "decode_ms": per_frame(self.decode_s),
            "infer_ms": per_frame(self.infer_s),
            "e2e_fps": round(steady / self.wall_s, 2) if steady > 0 and self.wall_s else None,
        }


@dataclass
class RunConfig:
    video: Path
    model: str
    size: str
    runtime: str
    imgsz: int
    stride: int
    max_frames: int | None
    out_dir: Path
    models_dir: Path

    @property
    def run_name(self) -> str:
        return f"{self.video.stem}.{self.model}-{self.size}.{self.runtime}.{self.imgsz}"


def validate(cfg: RunConfig) -> None:
    if cfg.model == "rtmpose" and cfg.runtime != "openvino":
        raise ValueError("rtmpose is only available as ONNX, run it with --runtime openvino")
    if cfg.model == "rtmpose" and cfg.size not in RTMPOSE_BODY7:
        raise ValueError(f"rtmpose sizes: {sorted(RTMPOSE_BODY7)}")
    if cfg.stride < 1:
        raise ValueError("--stride must be >= 1")


def ultralytics_device(runtime: str, ov_device: str | None) -> str:
    """Ultralytics device string: `xpu:0`, or `intel:<ov device>` to pin OpenVINO to one GPU."""
    if runtime == "xpu":
        return "xpu:0"
    if not ov_device:
        raise ValueError("no OpenVINO device to run on")
    return f"intel:{ov_device.lower()}"


def _num(v: float, ndigits: int) -> float | None:
    v = float(v)
    return round(v, ndigits) if math.isfinite(v) else None


def frame_record(
    frame: int,
    t_ms: float,
    boxes: Sequence[Sequence[float]],
    box_scores: Sequence[float],
    keypoints: Sequence[Sequence[Sequence[float]]],
    keypoint_scores: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """One output line. Keypoints are [x, y, score_raw]; non-finite values become null, never a guess."""
    if not (len(boxes) == len(box_scores) == len(keypoints) == len(keypoint_scores)):
        raise ValueError("boxes, scores and keypoints must describe the same people")
    players = []
    for box, box_score, kpts, kpt_scores in zip(boxes, box_scores, keypoints, keypoint_scores):
        players.append({
            "box": [_num(v, 1) for v in box],
            "box_score_raw": _num(box_score, 4),
            "keypoints": [[_num(x, 1), _num(y, 1), _num(s, 4)] for (x, y), s in zip(kpts, kpt_scores)],
        })
    return {"frame": frame, "t_ms": round(t_ms, 1), "players": players}


def iter_frames(video: Path, stride: int, max_frames: int | None) -> Iterator[tuple[int, float, Any]]:
    """(frame index, time in ms, BGR image) for every `stride`-th frame. Skipped frames are grabbed, not decoded."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    index = emitted = 0
    try:
        while (max_frames is None or emitted < max_frames) and cap.grab():
            if index % stride == 0:
                ok, image = cap.retrieve()
                if not ok:
                    break
                yield index, (1000 * index / fps if fps else 0.0), image
                emitted += 1
            index += 1
    finally:
        cap.release()


def video_info(video: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def load_yolo(name: str, cfg: RunConfig):
    """A YOLO11 model on the requested runtime. OpenVINO IR is exported once per imgsz and cached."""
    from ultralytics import YOLO

    weights = cfg.models_dir / f"{name}.pt"
    model = YOLO(str(weights))
    if cfg.runtime == "openvino":
        ir_dir = cfg.models_dir / f"{name}_openvino_{cfg.imgsz}"
        if not ir_dir.exists():
            exported = Path(model.export(format="openvino", half=True, imgsz=cfg.imgsz))
            exported.rename(ir_dir)
        model = YOLO(str(ir_dir), task="pose" if "pose" in name else "detect")
    return model


# A pipeline maps one BGR frame to (boxes, box scores, keypoints, keypoint scores), one entry per person.
Pipeline = Callable[[Any], tuple[list, list, list, list]]


def build_yolo_pose(cfg: RunConfig, device: str) -> Pipeline:
    model = load_yolo(f"yolo11{cfg.size}-pose", cfg)

    def infer(image):
        result = model.predict(image, device=device, imgsz=cfg.imgsz, conf=MIN_BOX_SCORE,
                               half=cfg.runtime == "xpu", verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
            return [], [], [], []
        return (result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                result.keypoints.xy.cpu().tolist(), result.keypoints.conf.cpu().tolist())

    return infer


def build_rtmpose(cfg: RunConfig, device: str, ov_device: str) -> Pipeline:
    os.environ.setdefault("TORCH_HOME", str(cfg.models_dir))  # rtmlib caches ONNX files under torch hub
    from rtmlib import RTMPose

    detector = load_yolo("yolo11x", cfg)
    url, input_size = RTMPOSE_BODY7[cfg.size]
    pose = RTMPose(onnx_model=url, model_input_size=input_size, backend="openvino", device=ov_device)

    def infer(image):
        det = detector.predict(image, device=device, imgsz=cfg.imgsz, conf=MIN_BOX_SCORE,
                               classes=[PERSON_CLASS], verbose=False)[0]
        boxes, scores = det.boxes.xyxy.cpu().tolist(), det.boxes.conf.cpu().tolist()
        # rtmlib treats an empty box list as "whole image", which would invent a person.
        if not boxes:
            return [], [], [], []
        kpts, kpt_scores = pose(image, bboxes=boxes)
        return boxes, scores, kpts.tolist(), kpt_scores.tolist()

    return infer


def process(frames: Iterator[tuple[int, float, Any]], infer: Pipeline, timing: Timing) -> Iterator[dict[str, Any]]:
    """Run the pipeline over frames, filling `timing`. Frame 1 is timed apart from the steady state."""
    frames = iter(frames)
    wall_start = None
    while True:
        start = time.perf_counter()
        try:
            index, t_ms, image = next(frames)
        except StopIteration:
            break
        decoded = time.perf_counter()
        boxes, scores, kpts, kpt_scores = infer(image)
        done = time.perf_counter()
        timing.frames += 1
        if timing.frames == 1:
            timing.first_frame_s = done - start
            wall_start = done
        else:
            timing.decode_s += decoded - start
            timing.infer_s += done - decoded
        yield frame_record(index, t_ms, boxes, scores, kpts, kpt_scores)
    if wall_start is not None:
        timing.wall_s = time.perf_counter() - wall_start


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for pkg in ("torch", "openvino", "ultralytics", "rtmlib", "opencv-python"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def run(cfg: RunConfig) -> Path:
    validate(cfg)
    from env_check import OV_DEVICE_ENV, pick_ov_gpu

    ov_device = None
    if cfg.runtime == "openvino":
        import openvino as ov

        core = ov.Core()
        gpus = [{"device": d, "name": core.get_property(d, "FULL_DEVICE_NAME"),
                 "type": str(core.get_property(d, "DEVICE_TYPE"))}
                for d in core.available_devices if d.startswith("GPU")]
        ov_device = pick_ov_gpu(gpus, os.environ.get(OV_DEVICE_ENV))
        if ov_device is None:
            raise RuntimeError(f"no discrete OpenVINO GPU among {gpus}")
    device = ultralytics_device(cfg.runtime, ov_device)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{cfg.run_name}.jsonl"
    timing = Timing()
    meta = {
        "run": cfg.run_name, "video": str(cfg.video), "video_info": video_info(cfg.video),
        "model": cfg.model, "size": cfg.size, "runtime": cfg.runtime, "device": device,
        "imgsz": cfg.imgsz, "stride": cfg.stride, "keypoint_format": "coco17 [x, y, score_raw]",
        "scores_calibrated": False, "min_box_score": MIN_BOX_SCORE, "versions": versions(),
    }
    infer = build_yolo_pose(cfg, device) if cfg.model == "yolo11-pose" else build_rtmpose(cfg, device, ov_device)

    with out_path.open("w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for record in process(iter_frames(cfg.video, cfg.stride, cfg.max_frames), infer, timing):
            f.write(json.dumps(record) + "\n")
            if timing.frames % 100 == 0:
                print(f"{cfg.run_name}: {timing.frames} frames", flush=True)

    summary = {**meta, **timing.summary()}
    (cfg.out_dir / f"{cfg.run_name}.timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(timing.summary()))
    return out_path


def parse_args(argv: list[str] | None = None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--size", choices=SIZES, default="x")
    p.add_argument("--runtime", choices=RUNTIMES, required=True)
    p.add_argument("--imgsz", type=int, default=1280, help="detector input size; 4K is downscaled to this")
    p.add_argument("--stride", type=int, default=1, help="process every n-th frame")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("out/pose"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    a = p.parse_args(argv)
    return RunConfig(a.video, a.model, a.size, a.runtime, a.imgsz, a.stride, a.max_frames, a.out_dir, a.models_dir)


def main(argv: list[str] | None = None) -> int:
    try:
        path = run(parse_args(argv))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
