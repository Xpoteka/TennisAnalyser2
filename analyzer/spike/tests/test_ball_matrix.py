import json

import ball_matrix
import ball_run


def test_every_combo_is_valid():
    for model, runtime, width, precision in ball_matrix.COMBOS:
        ball_run.validate(ball_run.BallConfig(None, model, runtime, width, precision, 0, None, None, None))


def test_matrix_keeps_going_after_a_failure(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "rally.mp4").write_text("")
    out = tmp_path / "out"

    def fake_run(cfg):
        if cfg.model == "tracknetv2":
            raise FileNotFoundError("weights missing")
        out.mkdir(exist_ok=True)
        (out / f"{cfg.run_name}.timing.json").write_text(json.dumps({"e2e_fps": 20.0}))

    monkeypatch.setattr(ball_run, "run", fake_run)
    code = ball_matrix.main(["--samples", str(samples), "--out-dir", str(out)])
    results = json.loads((out / "matrix.json").read_text())
    assert code == 1
    assert len(results) == len(ball_matrix.COMBOS)
    assert [r["ok"] for r in results].count(False) == 2


def test_labels_pick_the_frames_to_run(tmp_path):
    from test_labels import CVAT_VIDEO_XML

    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "rally.ball.xml").write_text(CVAT_VIDEO_XML)
    rally, other = tmp_path / "rally.mp4", tmp_path / "other.mp4"
    assert ball_matrix.ranges([rally, other], labels, 0, None) == [(rally, 100, 5), (rally, 900, 1)]
    assert ball_matrix.ranges([rally], None, 7, 50) == [(rally, 7, 50)]
