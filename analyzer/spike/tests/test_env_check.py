import json

import env_check
from env_check import Report, decide


def healthy() -> Report:
    return Report(
        host={"render_nodes": ["/dev/dri/renderD128"], "render_access": True},
        torch={"xpu_available": True, "resnet_speedup": 12.0},
        openvino={"devices": ["CPU", "GPU"], "resnet_speedup": 15.0},
        parity={"rel_error": 0.004},
    )


def test_healthy_report_is_go():
    assert decide(healthy()) == (True, [])


def test_missing_render_node_is_no_go():
    r = healthy()
    r.host = {"render_nodes": [], "render_access": False}
    go, reasons = decide(r)
    assert not go
    assert "renderD" in reasons[0]


def test_unreadable_render_node_points_at_group_add():
    r = healthy()
    r.host["render_access"] = False
    _, reasons = decide(r)
    assert "group_add" in reasons[0]


def test_xpu_slower_than_threshold_is_no_go():
    r = healthy()
    r.torch["resnet_speedup"] = 1.1
    go, reasons = decide(r)
    assert not go and reasons[0].startswith("torch:")


def test_openvino_without_gpu_is_no_go():
    r = healthy()
    r.openvino = {"devices": ["CPU"]}
    go, reasons = decide(r)
    assert not go and "no GPU device" in reasons[0]


def test_parity_error_too_large_is_no_go():
    r = healthy()
    r.parity = {"rel_error": 0.2}
    assert not decide(r)[0]


def test_errors_are_reported_per_section():
    r = healthy()
    r.torch = {"error": "ImportError: no torch"}
    r.parity = {"error": "skipped"}
    _, reasons = decide(r)
    assert [x.split(":")[0] for x in reasons] == ["torch", "parity"]


def test_time_per_iter_returns_median_and_syncs():
    calls = {"fn": 0, "sync": 0}
    median = env_check.time_per_iter(lambda: calls.__setitem__("fn", calls["fn"] + 1),
                                     lambda: calls.__setitem__("sync", calls["sync"] + 1))
    assert median >= 0
    assert calls["fn"] == env_check.WARMUP_ITERS + env_check.TIMED_ITERS
    assert calls["sync"] == env_check.TIMED_ITERS + 1


def test_main_writes_report_and_exit_code(tmp_path, monkeypatch):
    h = healthy()
    monkeypatch.setattr(env_check, "check_host", lambda: h.host)
    monkeypatch.setattr(env_check, "check_torch", lambda: h.torch)
    monkeypatch.setattr(env_check, "check_openvino", lambda: (h.openvino, h.parity))
    out = tmp_path / "env_check.json"
    assert env_check.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text())["go"] is True


def test_main_survives_a_crashing_section(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("no level zero")

    monkeypatch.setattr(env_check, "check_host", lambda: healthy().host)
    monkeypatch.setattr(env_check, "check_torch", boom)
    monkeypatch.setattr(env_check, "check_openvino", boom)
    out = tmp_path / "env_check.json"
    assert env_check.main(["--out", str(out)]) == 1
    report = json.loads(out.read_text())
    assert "no level zero" in report["torch"]["error"]
    assert report["parity"]["error"].startswith("skipped")
