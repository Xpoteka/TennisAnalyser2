import eval_ball as E
from labels import BallLabels


def test_tolerance_scales_with_width():
    assert E.tolerance(1280) == 4 and E.tolerance(3840) == 12


def rec(x=None, y=None, score=1.0):
    return {"ball": None if x is None else {"x": x, "y": y, "score_raw": score}}


def test_score_frames_classifies_every_case():
    labels = BallLabels(points={0: (100, 100), 1: (100, 100), 2: (100, 100)}, segments=[(0, 4)])
    records = {0: rec(105, 100), 1: rec(500, 500), 2: rec(), 3: rec(10, 10), 4: rec()}
    results, missing = E.score_frames("c", "low", labels, records, 3840)
    assert missing == 0
    c = E.counts(results)
    assert (c["tp"], c["fp"], c["fn"]) == (1, 2, 2)
    assert c["invented"] == 1 and c["invented_rate"] == 0.5


def test_missing_frames_are_counted_not_scored():
    labels = BallLabels(points={0: (1, 1)}, segments=[(0, 2)])
    results, missing = E.score_frames("c", "low", labels, {0: rec(1, 1)}, 1280)
    assert (len(results), missing) == (1, 2)


def test_evaluate_combo_calibrates_on_one_half_and_tests_on_the_other():
    labels = BallLabels(points={f: (100, 100) for f in range(240) if f % 4}, segments=[(0, 239)])
    # high scores land on the ball, low scores are made up
    records = {f: rec(100, 100, 5.0) if f % 4 else rec(900, 900, 0.2) for f in range(240)}
    results, _ = E.score_frames("c", "low", labels, records, 1280)
    report = E.evaluate_combo(results)
    g = report["groups"]["low"]
    assert report["calibration"] is not None
    assert g["fact"]["coverage"] == 1.0 and g["fact"]["precision"] == 1.0
    assert g["fact"]["invented_rate"] == 0.0
    assert g["pass"]
    assert set(report["groups"]) == {"low", "all"}


def test_no_calibration_means_nothing_is_shown_as_fact():
    labels = BallLabels(points={f: (1, 1) for f in range(200)}, segments=[(0, 199)])
    results, _ = E.score_frames("c", "low", labels, {f: rec(1, 1) for f in range(200)}, 1280)
    report = E.evaluate_combo(results)
    assert report["calibration"] is None and "note" in report
    assert report["groups"]["all"]["fact"]["shown"] == 0
    assert not report["groups"]["all"]["pass"]
