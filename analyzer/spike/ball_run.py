"""Phase 0, step 4: ball tracking on a sample clip, one model x runtime per run.

    python ball_run.py --video samples/rally.mp4 --model wasb --runtime openvino --start-frame 1200 --max-frames 500

Models:
  wasb        HRNet heatmap model from WASB-SBDT (BMVC 2023), tennis weights
  tracknetv2  TrackNetV2 (U-Net heatmap model) as re-trained by the WASB authors, tennis weights
  yolo-ball   baseline: YOLO11x "sports ball" class, one frame at a time

wasb and tracknetv2 see 3 consecutive frames at once and predict a heatmap for each. A heatmap is
thresholded into blobs; each blob is a candidate ball with `score_raw` = summed heatmap inside it
(the WASB postprocessing). Per frame the highest-scoring candidate within reach of the previous ball
is chosen. Weights come from Google Drive and must be downloaded once into <models-dir>/wasb/
(see README).

Output, under --out-dir: <run>.jsonl (line 1 = metadata, then one line per frame) and <run>.timing.json.
A frame without a candidate has `"ball": null`. Nothing is interpolated or predicted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from pose_run import load_yolo, ultralytics_device, versions

MODELS = ("wasb", "tracknetv2", "yolo-ball")
RUNTIMES = ("xpu", "openvino")
PRECISIONS = ("fp32", "fp16")
INPUT_WIDTHS = (512, 1024, 1536)  # 16:9, so heights 288 / 576 / 864

WEIGHT_FILES = {"wasb": "wasb_tennis_best.pth.tar", "tracknetv2": "tracknetv2_tennis_best.pth.tar"}
FRAMES_IN = 3
HEATMAP_THRESHOLD = 0.5  # WASB tennis config: postprocessor.score_threshold
# WASB's tracker rejects candidates farther than 300 px from the last ball, on 1280-wide video.
MAX_DISP_AT_1280 = 300.0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPORTS_BALL_CLASS = 32
YOLO_BALL_MIN_SCORE = 0.05

# WASB src/configs/model/wasb.yaml
WASB_HRNET_CFG = {
    "frames_in": 3, "frames_out": 3, "out_scales": [0],
    "MODEL": {"EXTRA": {
        "FINAL_CONV_KERNEL": 1, "PRETRAINED_LAYERS": ["*"],
        "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
        "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [1],
                   "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
        "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2],
                   "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
        "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2],
                   "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM"},
        "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2, 2],
                   "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": "SUM"},
        "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
    }, "INIT_WEIGHTS": True},
}


@dataclass
class BallConfig:
    video: Path
    model: str
    runtime: str
    inp_width: int
    precision: str
    start_frame: int
    max_frames: int | None
    out_dir: Path
    models_dir: Path
    imgsz: int = 1280  # yolo-ball detector input size

    @property
    def inp_height(self) -> int:
        return self.inp_width * 9 // 16

    @property
    def run_name(self) -> str:
        size = self.imgsz if self.model == "yolo-ball" else self.inp_width
        return f"{self.video.stem}.f{self.start_frame}.{self.model}.{self.runtime}.{size}.{self.precision}"


def validate(cfg: BallConfig) -> None:
    if cfg.inp_width not in INPUT_WIDTHS:
        raise ValueError(f"--inp-width must be one of {INPUT_WIDTHS}")
    if cfg.start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if cfg.max_frames is not None and cfg.max_frames < FRAMES_IN:
        raise ValueError(f"--max-frames must be >= {FRAMES_IN}")


# ---------- geometry: the same centred, aspect-preserving scaling WASB uses ----------

def input_transform(width: int, height: int, inp_w: int, inp_h: int) -> tuple[float, float, float]:
    """(k, tx, ty) so that input = k * original + t. Scales the longer side like WASB's get_transform."""
    k = inp_w / max(width, height)
    return k, inp_w / 2 - k * width / 2, inp_h / 2 - k * height / 2


def to_original(x: float, y: float, transform: tuple[float, float, float]) -> tuple[float, float]:
    k, tx, ty = transform
    return (x - tx) / k, (y - ty) / k


# ---------- heatmap decoding (WASB TracknetV2Postprocessor, blob_det_method=concomp, use_hm_weight=True) ----------

def detect_blobs(heatmap, threshold: float = HEATMAP_THRESHOLD) -> list[tuple[float, float, float, float]]:
    """Candidates as (x, y, score_raw, peak) in heatmap pixels. score_raw = summed heatmap in the blob."""
    import cv2
    import numpy as np

    if float(heatmap.max()) <= threshold:
        return []
    _, mask = cv2.threshold(heatmap.astype(np.float32), threshold, 1, cv2.THRESH_BINARY)
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    blobs = []
    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        ws = heatmap[ys, xs]
        total = float(ws.sum())
        blobs.append((float((xs * ws).sum() / total), float((ys * ws).sum() / total), total, float(ws.max())))
    return blobs


def select_ball(candidates: Sequence[tuple[float, float, float, float]], previous: tuple[float, float] | None,
                max_disp: float) -> tuple[float, float, float, float] | None:
    """WASB's online tracker: drop candidates too far from the last ball, keep the highest score."""
    if previous is not None:
        candidates = [c for c in candidates if math.dist(c[:2], previous) < max_disp]
    return max(candidates, key=lambda c: c[2], default=None)


def windows(frames: Iterable[Any], size: int = FRAMES_IN) -> Iterator[tuple[list[Any], int]]:
    """Non-overlapping windows of `size` frames (WASB step=3) as (window, first index to emit).

    A short tail is covered by a window of the last `size` frames that emits only the new ones,
    so every frame gets exactly one prediction.
    """
    recent: deque = deque(maxlen=size)
    pending = 0
    for frame in frames:
        recent.append(frame)
        pending += 1
        if pending == size:
            yield list(recent), 0
            pending = 0
    if pending and len(recent) == size:
        yield list(recent), size - pending


def ball_record(frame: int, t_ms: float, chosen, candidates) -> dict[str, Any]:
    def num(v, nd):
        v = float(v)
        return round(v, nd) if math.isfinite(v) else None

    return {
        "frame": frame,
        "t_ms": round(t_ms, 1),
        "ball": None if chosen is None else {"x": num(chosen[0], 1), "y": num(chosen[1], 1),
                                             "score_raw": num(chosen[2], 3), "peak": num(chosen[3], 4)},
        "candidates": [[num(c[0], 1), num(c[1], 1), num(c[2], 3)] for c in candidates],
    }


# ---------- models ----------

def build_heatmap_model(name: str):
    import torch

    if name == "wasb":
        from third_party.wasb.hrnet import HRNet
        net = HRNet(WASB_HRNET_CFG)
    else:
        from third_party.wasb.unet2d import TrackNetV2
        net = TrackNetV2(FRAMES_IN * 3, FRAMES_IN, bilinear=True, halve_channel=False)

    class FirstScale(torch.nn.Module):
        """The WASB models return {scale: logits}; OpenVINO conversion wants a plain tensor."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)[0]

    return net, FirstScale(net)


def load_weights(net, path: Path) -> None:
    import torch

    if not path.exists():
        raise FileNotFoundError(f"{path} missing: download it from the WASB model zoo (see README)")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}  # saved from nn.DataParallel
    net.load_state_dict(state)
    net.eval()


# Maps a stacked (1, 9, H, W) float32 numpy batch to (3, H, W) heatmap probabilities.
HeatmapFn = Callable[[Any], Any]


def heatmap_runner(cfg: BallConfig, ov_device: str | None) -> HeatmapFn:
    import numpy as np
    import torch

    net, wrapped = build_heatmap_model(cfg.model)
    load_weights(net, cfg.models_dir / "wasb" / WEIGHT_FILES[cfg.model])

    if cfg.runtime == "xpu":
        model = wrapped.to("xpu").eval()

        def run(batch):
            with torch.inference_mode(), torch.autocast("xpu", dtype=torch.float16, enabled=cfg.precision == "fp16"):
                logits = model(torch.from_numpy(batch).to("xpu"))
            return torch.sigmoid(logits.float())[0].cpu().numpy()
        return run

    import openvino as ov

    example = torch.zeros(1, FRAMES_IN * 3, cfg.inp_height, cfg.inp_width)
    ov_model = ov.convert_model(wrapped.eval(), example_input=example)
    compiled = ov.Core().compile_model(ov_model, ov_device, {
        "INFERENCE_PRECISION_HINT": "f16" if cfg.precision == "fp16" else "f32"})

    def run(batch):
        logits = compiled(batch)[0][0]
        return 1.0 / (1.0 + np.exp(-logits))
    return run


def preprocess(image_bgr, cfg: BallConfig, transform):
    """BGR frame -> normalised RGB (3, H, W) float32 at the model input size."""
    import cv2
    import numpy as np

    k, tx, ty = transform
    matrix = np.array([[k, 0, tx], [0, k, ty]], dtype=np.float32)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.warpAffine(rgb, matrix, (cfg.inp_width, cfg.inp_height), flags=cv2.INTER_LINEAR)
    x = small.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return x.transpose(2, 0, 1)


def iter_consecutive(video: Path, start: int, max_frames: int | None) -> Iterator[tuple[int, float, Any]]:
    """Consecutive frames from `start`. Frames before it are grabbed one by one: seeking is not frame-exact."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    try:
        for _ in range(start):
            if not cap.grab():
                return
        index, emitted = start, 0
        while max_frames is None or emitted < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            yield index, (1000 * index / fps if fps else 0.0), image
            index += 1
            emitted += 1
    finally:
        cap.release()


def track_heatmap_model(cfg: BallConfig, ov_device: str | None, frames, timing: BallTiming, info) -> Iterator[dict]:
    import numpy as np

    run = heatmap_runner(cfg, ov_device)
    transform = input_transform(info["width"], info["height"], cfg.inp_width, cfg.inp_height)
    max_disp = MAX_DISP_AT_1280 * info["width"] / 1280
    previous = None
    for window, emit_from in windows(frames):
        start = time.perf_counter()
        batch = np.concatenate([preprocess(img, cfg, transform) for _, _, img in window])[None]
        heatmaps = run(batch)
        done = time.perf_counter()
        for (index, t_ms, _), heatmap in list(zip(window, heatmaps))[emit_from:]:
            candidates = [(*to_original(x, y, transform), s, p) for x, y, s, p in detect_blobs(heatmap)]
            chosen = select_ball(candidates, previous, max_disp)
            previous = chosen[:2] if chosen else None
            yield ball_record(index, t_ms, chosen, candidates)
        timing.add_window(done - start, len(window) - emit_from)


def track_yolo(cfg: BallConfig, device: str, frames, timing: BallTiming, info) -> Iterator[dict]:
    model = load_yolo("yolo11x", cfg)
    max_disp = MAX_DISP_AT_1280 * info["width"] / 1280
    previous = None
    for index, t_ms, image in frames:
        start = time.perf_counter()
        det = model.predict(image, device=device, imgsz=cfg.imgsz, conf=YOLO_BALL_MIN_SCORE,
                            classes=[SPORTS_BALL_CLASS], verbose=False)[0]
        timing.add_window(time.perf_counter() - start, 1)
        candidates = []
        for (x1, y1, x2, y2), score in zip(det.boxes.xyxy.cpu().tolist(), det.boxes.conf.cpu().tolist()):
            candidates.append(((x1 + x2) / 2, (y1 + y2) / 2, score, score))
        chosen = select_ball(candidates, previous, max_disp)
        previous = chosen[:2] if chosen else None
        yield ball_record(index, t_ms, chosen, candidates)


@dataclass
class BallTiming:
    """Inference time per window of up to 3 frames. The first window (compilation) is reported apart."""

    frames: int = 0
    warmup_frames: int = 0
    infer_s: float = 0.0
    wall_s: float = 0.0
    first_window_s: float | None = None
    _wall_start: float | None = None

    def add_window(self, seconds: float, frames: int) -> None:
        if self.first_window_s is None:
            self.first_window_s, self.warmup_frames = seconds, frames
            self._wall_start = time.perf_counter()
        else:
            self.infer_s += seconds
        self.frames += frames

    def finish(self) -> None:
        if self._wall_start is not None:
            self.wall_s = time.perf_counter() - self._wall_start

    def summary(self) -> dict[str, Any]:
        steady = self.frames - self.warmup_frames
        return {
            "frames": self.frames,
            "first_window_ms": round(1000 * self.first_window_s, 1) if self.first_window_s is not None else None,
            "infer_ms": round(1000 * self.infer_s / steady, 2) if steady > 0 else None,
            # includes decoding and preprocessing, which run interleaved with inference
            "e2e_fps": round(steady / self.wall_s, 2) if steady > 0 and self.wall_s else None,
        }


def run(cfg: BallConfig) -> Path:
    validate(cfg)
    if not cfg.video.is_file():
        raise FileNotFoundError(f"video not found: {cfg.video}")
    from env_check import OV_DEVICE_ENV, pick_ov_gpu
    from pose_run import video_info

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

    info = video_info(cfg.video)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{cfg.run_name}.jsonl"
    meta = {
        "run": cfg.run_name, "video": str(cfg.video), "video_info": info, "model": cfg.model,
        "runtime": cfg.runtime, "device": ov_device or "xpu:0", "precision": cfg.precision,
        "input": [cfg.inp_width, cfg.inp_height] if cfg.model != "yolo-ball" else [cfg.imgsz, cfg.imgsz],
        "start_frame": cfg.start_frame, "heatmap_threshold": HEATMAP_THRESHOLD,
        "ball_format": "x, y in original pixels; score_raw = summed heatmap in blob (yolo-ball: box score)",
        "scores_calibrated": False, "versions": versions(),
    }
    frames = iter_consecutive(cfg.video, cfg.start_frame, cfg.max_frames)
    timing = BallTiming()
    if cfg.model == "yolo-ball":
        records = track_yolo(cfg, ultralytics_device(cfg.runtime, ov_device), frames, timing, info)
    else:
        records = track_heatmap_model(cfg, ov_device, frames, timing, info)

    with out_path.open("w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for record in records:
            f.write(json.dumps(record) + "\n")
    timing.finish()
    summary = {**meta, **timing.summary()}
    (cfg.out_dir / f"{cfg.run_name}.timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(timing.summary()))
    return out_path


def parse_args(argv: list[str] | None = None) -> BallConfig:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--runtime", choices=RUNTIMES, required=True)
    p.add_argument("--inp-width", type=int, default=512, help=f"heatmap model input width, one of {INPUT_WIDTHS}")
    p.add_argument("--precision", choices=PRECISIONS, default="fp32")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=1280, help="yolo-ball detector input size")
    p.add_argument("--out-dir", type=Path, default=Path("out/ball"))
    p.add_argument("--models-dir", type=Path, default=Path("models"))
    a = p.parse_args(argv)
    return BallConfig(a.video, a.model, a.runtime, a.inp_width, a.precision, a.start_frame, a.max_frames,
                      a.out_dir, a.models_dir, a.imgsz)


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
