import pytest

import calibration as C


def test_split_alternates_by_second():
    assert C.split_of(0) == C.split_of(59) == "calibrate"
    assert C.split_of(60) == C.split_of(119) == "test"
    assert C.split_of(120) == "calibrate"


def test_isotonic_is_monotone_and_bounded():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    correct = [0, 0, 1, 0, 0, 1, 1, 1, 1, 1]
    cal = C.fit_isotonic(scores, correct)
    values = [cal(s) for s in [0.0, *scores, 5.0]]
    assert values == sorted(values)
    assert 0.0 <= values[0] and values[-1] == pytest.approx(1.0)


def test_isotonic_needs_both_outcomes():
    assert C.fit_isotonic([0.5, 0.6], [1, 1]) is None
    assert C.fit_isotonic([], []) is None


def test_reliability_and_ece():
    r = C.reliability([0.95, 0.95, 0.15, 0.15], [1, 0, 0, 0])
    assert [b["n"] for b in r["bins"]] == [2, 2]
    assert r["ece"] == pytest.approx(0.5 * 0.45 + 0.5 * 0.15)


def test_at_threshold_counts_facts():
    out = C.at_threshold([0.99, 0.97, 0.5, 0.96], [1, 1, 1, 0], n_targets=5)
    assert (out["shown"], out["correct"]) == (3, 2)
    assert out["coverage"] == 0.4 and out["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert out["precision_lower_95"] < out["precision"]


def test_wilson_lower_bound():
    assert C.wilson_lower(0, 0) is None
    assert C.wilson_lower(95, 100) == pytest.approx(0.8882, abs=1e-3)
