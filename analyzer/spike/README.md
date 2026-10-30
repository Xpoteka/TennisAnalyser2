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

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the GO/NO-GO logic and report writing and need no GPU.

## Versions

`openvino` is pinned in `requirements.txt`. `torch`/`torchvision` come unpinned from the PyTorch
XPU index until the first GO run; pin them to the versions recorded in `env_check.json` then.
