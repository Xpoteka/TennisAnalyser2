import json
import math
from pathlib import Path

import pytest

import pose_run
from pose_run import RunConfig, Timing, frame_record, process


def cfg(**overrides) -> RunConfig:
    values = dict(video=Path("samples/rally.mp4"), model="yolo11-pose", size="x", runtime="openvino",
                  imgsz=1280, stride=1, max_frames=None, out_dir=Path("out"), models_dir=Path("models"))
    values.update(overrides)
    return RunConfig(**values)


def test_run_name_identifies_model_runtime_and_input_size():
    assert cfg().run_name == "rally.yolo11-pose-x.openvino.1280"


def test_rtmpose_needs_openvino():
    with pytest.raises(ValueError, match="openvino"):
        pose_run.validate(cfg(model="rtmpose", runtime="xpu"))


def test_rtmpose_rejects_sizes_without_weights():
    with pytest.raises(ValueError, match="sizes"):
        pose_run.validate(cfg(model="rtmpose", size="l"))


def test_stride_must_be_positive():
    with pytest.raises(ValueError):
        pose_run.validate(cfg(stride=0))


def test_ultralytics_device_pins_the_openvino_gpu():
    assert pose_run.ultralytics_device("openvino", "GPU.1") == "intel:gpu.1"
    assert pose_run.ultralytics_device("xpu", None) == "xpu:0"
    with pytest.raises(ValueError):
        pose_run.ultralytics_device("openvino", None)


def test_frame_record_shape_and_rounding():
    rec = frame_record(7, 116.666, [[1.04, 2.0, 3.0, 4.0]], [0.91234], [[[10.04, 20.06], [11.0, 21.0]]], [[0.5, 0.25]])
    assert rec == {"frame": 7, "t_ms": 116.7, "players": [{
        "box": [1.0, 2.0, 3.0, 4.0], "box_score_raw": 0.9123,
        "keypoints": [[10.0, 20.1, 0.5], [11.0, 21.0, 0.25]]}]}
    json.dumps(rec)  # must be serialisable as one JSONL line


def test_frame_record_without_people_is_empty_not_guessed():
    assert frame_record(0, 0.0, [], [], [], [])["players"] == []


def test_frame_record_turns_non_finite_values_into_null():
    rec = frame_record(0, 0.0, [[0, 0, 1, 1]], [math.nan], [[[math.inf, 1.0]]], [[0.3]])
    assert rec["players"][0]["box_score_raw"] is None
    assert rec["players"][0]["keypoints"][0][0] is None


def test_frame_record_rejects_mismatched_people():
    with pytest.raises(ValueError):
        frame_record(0, 0.0, [[0, 0, 1, 1]], [0.9], [], [])


def test_process_times_first_frame_apart_and_keeps_frame_indices():
    frames = [(0, 0.0, "img0"), (2, 33.3, "img2"), (4, 66.7, "img4")]
    seen = []

    def infer(image):
        seen.append(image)
        return ([[0, 0, 1, 1]], [0.9], [[[0.5, 0.5]]], [[0.8]]) if image == "img2" else ([], [], [], [])

    timing = Timing()
    records = list(process(frames, infer, timing))
    assert seen == ["img0", "img2", "img4"]
    assert [r["frame"] for r in records] == [0, 2, 4]
    assert [len(r["players"]) for r in records] == [0, 1, 0]
    summary = timing.summary()
    assert summary["frames"] == 3
    assert summary["first_frame_ms"] is not None
    assert summary["e2e_fps"] is not None and summary["e2e_fps"] > 0


def test_timing_summary_with_a_single_frame_has_no_steady_state():
    t = Timing(frames=1, first_frame_s=0.5)
    assert t.summary() == {"frames": 1, "first_frame_ms": 500.0, "decode_ms": None, "infer_ms": None, "e2e_fps": None}


def test_main_reports_invalid_combination_without_traceback(capsys):
    code = pose_run.main(["--video", "x.mp4", "--model", "rtmpose", "--runtime", "xpu"])
    assert code == 2
    assert "openvino" in capsys.readouterr().err
