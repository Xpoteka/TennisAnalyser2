import json

import evaluate
from test_labels import CVAT_VIDEO_XML


def write_run(path, meta, records):
    with path.open("w") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")


def ball_meta(runtime):
    return {"run": f"rally.{runtime}", "video": "/work/samples/rally.mp4", "model": "wasb", "runtime": runtime,
            "input": [512, 288], "precision": "fp32", "video_info": {"width": 3840, "height": 2160}}


def test_end_to_end_writes_report(tmp_path):
    runs, labels, out = tmp_path / "out", tmp_path / "labels", tmp_path / "eval"
    (runs / "ball").mkdir(parents=True)
    labels.mkdir()
    (labels / "rally.ball.xml").write_text(CVAT_VIDEO_XML)
    (labels / "clips.json").write_text(json.dumps({"rally": {"height": "high"}}))
    for rt, dx in (("xpu", 0), ("openvino", 3)):
        recs = [{"frame": f, "ball": {"x": 1000.5 + dx, "y": 500.0, "score_raw": 3.0}} for f in range(95, 110)]
        write_run(runs / "ball" / f"rally.{rt}.jsonl", ball_meta(rt), recs)

    assert evaluate.main(["--runs", str(runs), "--labels", str(labels), "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert set(report["ball"]["combos"]) == {"wasb.xpu.512.fp32", "wasb.openvino.512.fp32"}
    assert "high" in report["ball"]["combos"]["wasb.xpu.512.fp32"]["groups"]
    assert report["ball"]["frames_not_run"]["wasb.xpu.512.fp32"] == {"rally": 1}  # frame 900
    parity = report["ball"]["parity"]["rally.wasb.512.fp32"]
    assert parity["mean_px"] == 3.0 and parity["found_agree"] == 1.0
    assert "## Ball" in (out / "REPORT.md").read_text()


def test_pose_parity_matches_by_box():
    kp = [[100.0, 100.0, 0.9]] * 17
    a = {0: {"players": [{"box": [0, 0, 100, 200], "keypoints": kp}]}}
    b = {0: {"players": [{"box": [2, 0, 102, 200], "keypoints": [[101.0, 100.0, 0.9]] * 17}]}}
    p = evaluate.pose_parity(a, b)
    assert p["mean_px"] == 1.0 and p["pass"]


def test_missing_labels_folder_is_an_error(tmp_path):
    assert evaluate.main(["--labels", str(tmp_path / "nope")]) == 2
