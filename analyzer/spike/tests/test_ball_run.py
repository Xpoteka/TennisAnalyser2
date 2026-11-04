from pathlib import Path

import numpy as np
import pytest

import ball_run
from ball_run import BallConfig, BallTiming, detect_blobs, input_transform, select_ball, to_original, windows


def cfg(**overrides) -> BallConfig:
    values = dict(video=Path("samples/rally.mp4"), model="wasb", runtime="openvino", inp_width=512,
                  precision="fp32", start_frame=1200, max_frames=500, out_dir=Path("out"), models_dir=Path("models"))
    values.update(overrides)
    return BallConfig(**values)


def test_run_name_and_input_size():
    c = cfg(inp_width=1024)
    assert c.inp_height == 576
    assert c.run_name == "rally.f1200.wasb.openvino.1024.fp32"


def test_validate_rejects_unknown_width_and_tiny_runs():
    with pytest.raises(ValueError):
        ball_run.validate(cfg(inp_width=600))
    with pytest.raises(ValueError):
        ball_run.validate(cfg(max_frames=2))


def test_input_transform_round_trips_4k():
    t = input_transform(3840, 2160, 512, 288)
    assert t[0] == pytest.approx(512 / 3840)
    assert to_original(256, 144, t) == pytest.approx((1920, 1080))
    assert to_original(0, 0, t) == pytest.approx((0, 0))


def test_input_transform_letterboxes_other_aspect_ratios():
    k, tx, ty = input_transform(1440, 1080, 512, 288)  # 4:3 into 16:9: the longer side sets the scale
    assert (k, tx) == (pytest.approx(512 / 1440), pytest.approx(0))
    assert ty < 0  # top and bottom are cut, like WASB's get_transform


def test_detect_blobs_weighted_centre_and_score():
    hm = np.zeros((20, 30), dtype=np.float32)
    hm[5, 10] = 0.9
    hm[5, 11] = 0.7
    hm[15, 25] = 0.6
    blobs = sorted(detect_blobs(hm), key=lambda b: -b[2])
    assert len(blobs) == 2
    x, y, score, peak = blobs[0]
    assert x == pytest.approx((10 * 0.9 + 11 * 0.7) / 1.6)
    assert y == pytest.approx(5)
    assert score == pytest.approx(1.6)
    assert peak == pytest.approx(0.9)


def test_detect_blobs_below_threshold_finds_nothing():
    assert detect_blobs(np.full((10, 10), 0.4, dtype=np.float32)) == []


def test_select_ball_prefers_score_but_respects_max_disp():
    candidates = [(100, 100, 2.0, 0.9), (900, 900, 5.0, 0.9)]
    assert select_ball(candidates, None, 300)[:2] == (900, 900)
    assert select_ball(candidates, (110, 90), 300)[:2] == (100, 100)
    assert select_ball(candidates, (2000, 2000), 300) is None
    assert select_ball([], None, 300) is None


def test_windows_cover_every_frame_exactly_once():
    for n in range(3, 12):
        emitted = []
        for window, start in windows(range(n)):
            assert len(window) == 3
            emitted += window[start:]
        assert emitted == list(range(n))


def test_windows_with_fewer_frames_than_the_model_needs_yield_nothing():
    assert list(windows(range(2))) == []


def test_ball_record_null_when_nothing_found():
    rec = ball_run.ball_record(5, 83.33, None, [])
    assert rec == {"frame": 5, "t_ms": 83.3, "ball": None, "candidates": []}


def test_ball_record_rounds_and_keeps_all_candidates():
    rec = ball_run.ball_record(1, 16.7, (10.04, 20.06, 1.23456, 0.87654), [(10.04, 20.06, 1.23456, 0.87654), (5, 5, 0.6, 0.6)])
    assert rec["ball"] == {"x": 10.0, "y": 20.1, "score_raw": 1.235, "peak": 0.8765}
    assert rec["candidates"] == [[10.0, 20.1, 1.235], [5.0, 5.0, 0.6]]


def test_ball_timing_excludes_first_window():
    t = BallTiming()
    t.add_window(2.0, 3)
    t.add_window(0.03, 3)
    t.add_window(0.03, 3)
    t.wall_s = 0.1
    s = t.summary()
    assert s["frames"] == 9
    assert s["first_window_ms"] == 2000.0
    assert s["infer_ms"] == pytest.approx(10.0)
    assert s["e2e_fps"] == pytest.approx(60.0)


def test_preprocess_shape_and_normalisation():
    c = cfg()
    img = np.full((2160, 3840, 3), 255, dtype=np.uint8)
    x = ball_run.preprocess(img, c, input_transform(3840, 2160, 512, 288))
    assert x.shape == (3, 288, 512) and x.dtype == np.float32
    assert x[0, 100, 100] == pytest.approx((1 - 0.485) / 0.229, rel=1e-5)


def test_main_reports_missing_video_without_traceback(capsys, tmp_path):
    code = ball_run.main(["--video", str(tmp_path / "nope.mp4"), "--model", "wasb", "--runtime", "xpu",
                          "--out-dir", str(tmp_path)])
    assert code == 2
    assert "error" in capsys.readouterr().err
