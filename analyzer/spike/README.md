# Phase 0 GPU spike

Throwaway spike code that answers one question: does the Intel Arc Pro B70 run pose and ball
tracking well enough, and on which runtime (PyTorch XPU or OpenVINO)? The plan with candidates,
metrics and pass marks lives in the project's spike plan doc. Results go into
`docs/decisions.md` when the spike is done.

## Step 1: environment check

`env_check.py` checks, from inside the container, that:

1. `/dev/dri/renderD*` is mapped and accessible,
2. PyTorch sees the `xpu` device and a ResNet-50 runs at least 3x faster than on the CPU,
3. OpenVINO sees a `GPU` device and the same model runs at least 3x faster than on the CPU,
4. OpenVINO GPU output matches PyTorch CPU output (relative error at most 5%, because OpenVINO GPU defaults to FP16).

OpenVINO lists an integrated GPU as another `GPU.n` device, so the check uses the first **discrete**
GPU and never falls back to an iGPU. Every GPU it sees is listed under `openvino.gpus`. To force a
device, pass `-e OV_DEVICE=GPU.1`.

It prints a JSON report plus `GO` or `NO-GO` with reasons, writes the report to
`/work/out/env_check.json`, and exits 1 on NO-GO. The model has random weights; this measures
the hardware path, not accuracy.

### Run on TrueNAS

CI publishes the image to `ghcr.io/xpoteka/tennis-analyzer-spike` (see "Image releases" below),
so TrueNAS only pulls it. On the TrueNAS shell, from this folder of the repo:

```bash
getent group render                     # note the gid
RENDER_GID=<gid> SPIKE_DATA=/mnt/<pool>/tennis sudo -E docker compose up
cat /mnt/<pool>/tennis/spike-out/env_check.json
```

`SPIKE_DATA` must contain a `samples/` folder (mounted read-only, used from step 3 on) and gets a
`spike-out/` folder for results. Add `--build` to build the image on TrueNAS instead of pulling it.
A TrueNAS custom app with the same `image`, `devices`, `group_add` and `volumes` settings as
`compose.yaml` works too.

If the repository is private, its packages are private too: log in once with
`sudo docker login ghcr.io -u <github user>` and a personal access token with `read:packages`.

## Image releases

`.github/workflows/analyzer-spike-image.yml` runs the tests and builds the image:

| Trigger | Published tags |
| --- | --- |
| Pull request | none (build check only) |
| Push to `main` touching `analyzer/spike/` | `main`, `sha-<short commit>` |
| Git tag `v1.2.3` | `1.2.3`, `1.2`, `latest` |

Pick a tag on TrueNAS with `SPIKE_TAG=<tag>`; the default is `main`.

### Known-good TrueNAS setup

As a custom app: enable **Use all GPUs** (TrueNAS refuses `/dev/dri` as a manual device). The app
runs as `apps` (568:568), so `spike-out/` and `models/` must be writable by that user.

## Step 3: pose runs

`pose_run.py` runs one pose model on one clip and writes per-frame JSON:

```bash
python pose_run.py --video /work/samples/rally.mp4 --model yolo11-pose --size x --runtime openvino \
    --out-dir /work/out/pose --models-dir /work/models --max-frames 600
```

| Option | Values |
| --- | --- |
| `--model` | `yolo11-pose` (one-stage), `rtmpose` (YOLO11x finds people, RTMPose body7 puts keypoints on each) |
| `--size` | `m`, `l`, `x` for YOLO11-Pose; `m`, `x` for RTMPose |
| `--runtime` | `xpu` or `openvino`; RTMPose is ONNX-only, so OpenVINO only |
| `--imgsz` | detector input size, default 1280 (4K frames are scaled down to this) |
| `--stride`, `--max-frames` | sample fewer frames for quick runs |

`pose_matrix.py` runs all six candidates on every video in `samples/` and writes `matrix.json`
with the FPS of each, or the error if one fails. In the TrueNAS app, set the command to:

```
python pose_matrix.py --samples /work/samples --out-dir /work/out/pose --models-dir /work/models --max-frames 600
```

Output per run, in `out/pose/`:

- `<clip>.<model>-<size>.<runtime>.<imgsz>.jsonl`: line 1 is the run metadata, then one line per
  frame: `{"frame", "t_ms", "players": [{"box", "box_score_raw", "keypoints": [[x, y, score_raw] x17]}]}`.
  A frame without people has `"players": []`. Scores are raw model outputs, not calibrated (step 6).
- `<run>.timing.json`: first-frame time (compilation), then mean decode and inference ms per frame and FPS.

The first run downloads the weights (Ultralytics, OpenMMLab) into `models/` and exports the
OpenVINO versions there, so later runs start quickly.

## Step 4: ball runs

`ball_run.py` runs one ball tracker on consecutive frames of one clip:

```bash
python ball_run.py --video /work/samples/rally.mp4 --model wasb --runtime openvino \
    --start-frame 1200 --max-frames 500 --out-dir /work/out/ball --models-dir /work/models
```

| Option | Values |
| --- | --- |
| `--model` | `wasb` (HRNet, BMVC 2023), `tracknetv2` (WASB authors' re-trained TrackNetV2), `yolo-ball` (YOLO11x "sports ball", baseline) |
| `--runtime` | `xpu` or `openvino` |
| `--inp-width` | heatmap model input: `512` (the models' training size), `1024`, `1536`; height follows 16:9 |
| `--precision` | `fp32` (default) or `fp16` |
| `--start-frame`, `--max-frames` | the stretch to process, e.g. one labeled run |

`ball_matrix.py` runs seven combinations on every clip and writes `out/ball/matrix.json`.

**Weights (one-time manual download).** The WASB tennis weights are on Google Drive, which
doesn't allow reliable scripted downloads. Download both files and put them in `<models>/wasb/`
(on TrueNAS `/mnt/HDDs/tennis/models/wasb/`):

| File | Link |
| --- | --- |
| `wasb_tennis_best.pth.tar` | https://drive.google.com/file/d/14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z |
| `tracknetv2_tennis_best.pth.tar` | https://drive.google.com/file/d/1BCnmvDX-LZpbkk4vlMEXMm-uzCoqJzDx |

The model code is vendored, unchanged, in `third_party/wasb/` (MIT licence, see the file there).

Output per run, in `out/ball/`:

- `<clip>.f<start>.<model>.<runtime>.<width>.<precision>.jsonl`: line 1 is the metadata, then one
  line per frame: `{"frame", "t_ms", "ball": {"x", "y", "score_raw", "peak"} or null, "candidates": [[x, y, score_raw], ...]}`.
  `score_raw` is the summed heatmap inside the chosen blob (WASB's own score), not a probability.
  `ball` is null when no candidate passes the heatmap threshold; nothing is interpolated.
- `<run>.timing.json`: first-window time (compilation), mean inference ms per frame and FPS.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the GO/NO-GO logic and report writing and need no GPU.

## Versions

`openvino` is pinned in `requirements.txt`. `torch`/`torchvision` come unpinned from the PyTorch
XPU index until the first GO run; pin them to the versions recorded in `env_check.json` then.
