import json

import pose_matrix
import pose_run


def test_find_videos_ignores_other_files(tmp_path):
    for name in ("b.MP4", "a.mov", "labels.json", "notes.txt"):
        (tmp_path / name).write_text("")
    assert [p.name for p in pose_matrix.find_videos(tmp_path)] == ["a.mov", "b.MP4"]


def test_every_combo_is_valid():
    for model, size, runtime in pose_matrix.COMBOS:
        pose_run.validate(pose_run.RunConfig(
            None, model, size, runtime, 1280, 1, None, None, None))


def test_matrix_keeps_going_after_a_failure(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "rally.mp4").write_text("")
    out = tmp_path / "out"

    def fake_run(cfg):
        if cfg.model == "rtmpose":
            raise RuntimeError("compile failed")
        out.mkdir(exist_ok=True)
        (out / f"{cfg.run_name}.timing.json").write_text(json.dumps({"e2e_fps": 5.0}))

    monkeypatch.setattr(pose_run, "run", fake_run)
    code = pose_matrix.main(["--samples", str(samples), "--out-dir", str(out)])
    results = json.loads((out / "matrix.json").read_text())
    assert code == 1
    assert len(results) == len(pose_matrix.COMBOS)
    assert [r["ok"] for r in results].count(False) == 2


def test_matrix_without_videos_is_an_error(tmp_path):
    assert pose_matrix.main(["--samples", str(tmp_path)]) == 2
