import eval_pose as E
from labels import PosePlayer


def player(cx, cy, size=200, v=2):
    kpts = [(cx, cy - size / 2 + i * size / 16, v) for i in range(17)]
    return PosePlayer(kpts, (cx - 20, cy - size / 2, 40, size))


def person(p: PosePlayer, dx=0.0, score=0.9):
    return {"box": [0, 0, 1, 1], "box_score_raw": 0.9, "keypoints": [[x + dx, y, score] for x, y, _ in p.keypoints]}


def test_side_uses_the_frame_half():
    assert E.side(player(500, 1600), 2160) == "near"
    assert E.side(player(500, 400), 2160) == "far"


def test_match_players_pairs_each_label_with_its_person():
    near, far = player(500, 1600), player(2000, 400, size=80)
    matched = E.match_players([near, far], [person(far, 1), person(near, 2)])
    assert matched[0][0][0] == near.keypoints[0][0] + 2
    assert matched[1][0][0] == far.keypoints[0][0] + 1


def test_match_rejects_a_different_person():
    assert E.match_players([player(500, 1600)], [person(player(3000, 400))]) == [None]


def test_pck_uses_player_size_and_skips_unlabeled():
    near = player(500, 1600, size=200)  # PCK distance = 10 px
    near.keypoints[0] = (0, 0, 0)
    labels = {0: [near]}
    results, _ = E.score_frames("c", "low", labels, {0: {"players": [person(near, dx=9)]}}, 2160)
    assert len(results) == 16 and all(r.correct for r in results)
    results, _ = E.score_frames("c", "low", labels, {0: {"players": [person(near, dx=11)]}}, 2160)
    assert not any(r.correct for r in results)


def test_unmatched_player_counts_as_missed():
    results, _ = E.score_frames("c", "low", {0: [player(500, 1600)]}, {0: {"players": []}}, 2160)
    assert len(results) == 17 and not any(r.predicted for r in results)


def test_evaluate_combo_reports_near_arms():
    labels = {f: [player(500, 1600), player(2000, 400, size=80)] for f in range(0, 240, 2)}
    records = {f: {"players": [person(labels[f][0], dx=1, score=0.9), person(labels[f][1], dx=30, score=0.1)]}
               for f in labels}
    results = []
    for f in labels:
        rs, _ = E.score_frames("c", "low", {f: labels[f]}, records, 2160)
        results += rs
    report = E.evaluate_combo(results)
    g = report["groups"]["all"]
    assert g["near"]["pck_arms"] == 1.0 and g["far"]["pck_arms"] == 0.0
    assert g["near"]["arms_fact"]["coverage"] == 1.0
    assert g["pass"]
