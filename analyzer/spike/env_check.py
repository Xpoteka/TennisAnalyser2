"""Phase 0, step 1: does the Arc GPU work inside this container, on both runtimes?

Checks, in order:
  1. Host side as seen from the container: kernel, /dev/dri render nodes, access rights.
  2. PyTorch XPU: device visible, a matmul and a ResNet-50 forward pass run on it and beat the CPU.
  3. OpenVINO: GPU device visible, the same ResNet-50 compiles and runs on it.
  4. Parity: OpenVINO GPU output matches PyTorch CPU output on the same input.

Writes a JSON report and exits 0 on GO, 1 on NO-GO. ResNet-50 uses random weights:
this measures whether the hardware path works, not model accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# The GPU must be at least this much faster than the CPU, otherwise ops are probably
# falling back to the CPU or the device is not really being used.
MIN_GPU_SPEEDUP = 3.0
# OpenVINO GPU runs in FP16 by default, so compare against FP32 CPU with a loose bound.
MAX_PARITY_REL_ERROR = 0.05

RESNET_BATCH = 8
WARMUP_ITERS = 5
TIMED_ITERS = 20


@dataclass
class Report:
    host: dict[str, Any] = field(default_factory=dict)
    torch: dict[str, Any] = field(default_factory=dict)
    openvino: dict[str, Any] = field(default_factory=dict)
    parity: dict[str, Any] = field(default_factory=dict)
    go: bool = False
    reasons: list[str] = field(default_factory=list)


def decide(report: Report) -> tuple[bool, list[str]]:
    """Pure GO/NO-GO decision over a filled-in report. Every failure gives a reason."""
    reasons: list[str] = []
    if not report.host.get("render_nodes"):
        reasons.append("no /dev/dri/renderD* node in the container (check the devices: mapping)")
    elif not report.host.get("render_access"):
        reasons.append("render node not readable/writable (check group_add with the host render gid)")

    t = report.torch
    if t.get("error"):
        reasons.append(f"torch: {t['error']}")
    elif not t.get("xpu_available"):
        reasons.append("torch: torch.xpu.is_available() is False")
    elif t.get("resnet_speedup") is None or t["resnet_speedup"] < MIN_GPU_SPEEDUP:
        reasons.append(f"torch: XPU ResNet-50 speedup {t.get('resnet_speedup')} < {MIN_GPU_SPEEDUP}x over CPU")

    o = report.openvino
    if o.get("error"):
        reasons.append(f"openvino: {o['error']}")
    elif "GPU" not in " ".join(o.get("devices", [])):
        reasons.append(f"openvino: no GPU device (devices: {o.get('devices')})")
    elif o.get("resnet_speedup") is None or o["resnet_speedup"] < MIN_GPU_SPEEDUP:
        reasons.append(f"openvino: GPU ResNet-50 speedup {o.get('resnet_speedup')} < {MIN_GPU_SPEEDUP}x over CPU")

    p = report.parity
    if p.get("error"):
        reasons.append(f"parity: {p['error']}")
    elif p.get("rel_error") is None or p["rel_error"] > MAX_PARITY_REL_ERROR:
        reasons.append(f"parity: OpenVINO vs PyTorch relative error {p.get('rel_error')} > {MAX_PARITY_REL_ERROR}")

    return not reasons, reasons


def time_per_iter(fn: Callable[[], Any], sync: Callable[[], None] = lambda: None) -> float:
    """Median seconds per call after warm-up. sync() must block until queued GPU work is done."""
    for _ in range(WARMUP_ITERS):
        fn()
    sync()
    samples = []
    for _ in range(TIMED_ITERS):
        start = time.perf_counter()
        fn()
        sync()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def check_host() -> dict[str, Any]:
    dri = Path("/dev/dri")
    nodes = sorted(str(p) for p in dri.glob("renderD*")) if dri.exists() else []
    return {
        "kernel": platform.release(),
        "python": platform.python_version(),
        "render_nodes": nodes,
        "render_access": bool(nodes) and all(os.access(n, os.R_OK | os.W_OK) for n in nodes),
        "groups": os.getgroups(),
    }


def build_resnet(seed: int = 0):
    import torch
    import torchvision

    torch.manual_seed(seed)
    model = torchvision.models.resnet50(weights=None).eval()
    example = torch.randn(RESNET_BATCH, 3, 224, 224)
    return model, example


def check_torch() -> dict[str, Any]:
    import torch

    out: dict[str, Any] = {"version": torch.__version__, "xpu_available": torch.xpu.is_available()}
    if not out["xpu_available"]:
        return out

    out["devices"] = [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())]
    props = torch.xpu.get_device_properties(0)
    out["total_memory_mb"] = round(props.total_memory / 2**20)
    sync = torch.xpu.synchronize

    a = torch.randn(4096, 4096, device="xpu", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="xpu", dtype=torch.float16)
    matmul_s = time_per_iter(lambda: a @ b, sync)
    out["matmul_fp16_tflops"] = round(2 * 4096**3 / matmul_s / 1e12, 2)

    model, example = build_resnet()
    with torch.inference_mode():
        cpu_s = time_per_iter(lambda: model(example))
        model_xpu, example_xpu = model.to("xpu"), example.to("xpu")
        with torch.autocast("xpu", dtype=torch.float16):
            xpu_s = time_per_iter(lambda: model_xpu(example_xpu), sync)
    out["resnet_cpu_img_s"] = round(RESNET_BATCH / cpu_s, 1)
    out["resnet_xpu_fp16_img_s"] = round(RESNET_BATCH / xpu_s, 1)
    out["resnet_speedup"] = round(cpu_s / xpu_s, 1)
    return out


def check_openvino() -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import openvino as ov
    import torch

    core = ov.Core()
    out: dict[str, Any] = {"version": ov.get_version(), "devices": core.available_devices}
    gpus = [d for d in out["devices"] if d.startswith("GPU")]
    if not gpus:
        return out, {"error": "skipped, no OpenVINO GPU"}
    out["gpu_name"] = core.get_property(gpus[0], "FULL_DEVICE_NAME")

    model, example = build_resnet()
    ov_model = ov.convert_model(model, example_input=example)
    x = example.numpy()

    cpu = core.compile_model(ov_model, "CPU")
    gpu = core.compile_model(ov_model, gpus[0], {"PERFORMANCE_HINT": "THROUGHPUT"})
    cpu_s = time_per_iter(lambda: cpu(x))
    gpu_s = time_per_iter(lambda: gpu(x))  # synchronous call: returns when results are ready
    out["resnet_cpu_img_s"] = round(RESNET_BATCH / cpu_s, 1)
    out["resnet_gpu_img_s"] = round(RESNET_BATCH / gpu_s, 1)
    out["resnet_speedup"] = round(cpu_s / gpu_s, 1)

    with torch.inference_mode():
        reference = model(example).numpy()
    got = gpu(x)[0]
    rel = float(np.linalg.norm(got - reference) / np.linalg.norm(reference))
    parity = {"rel_error": round(rel, 4), "compared": "openvino GPU vs torch CPU fp32, ResNet-50 logits"}
    return out, parity


def run_section(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:  # report every failure instead of stopping at the first one
        return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("out/env_check.json"))
    args = parser.parse_args(argv)

    report = Report(host=check_host())
    report.torch = run_section(check_torch)
    ov_result = run_section(check_openvino)
    if isinstance(ov_result, tuple):
        report.openvino, report.parity = ov_result
    else:
        report.openvino = ov_result
        report.parity = {"error": "skipped, OpenVINO check failed"}
    report.go, report.reasons = decide(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2))
    print(f"\n{'GO' if report.go else 'NO-GO'}" + "".join(f"\n - {r}" for r in report.reasons))
    print(f"Report written to {args.out}")
    return 0 if report.go else 1


if __name__ == "__main__":
    sys.exit(main())
