import json

import pytest

import labels as L

CVAT_VIDEO_XML = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <track id="0" label="ball" source="manual">
    <points frame="100" keyframe="1" outside="0" occluded="0" points="1000.5,500.0" z_order="0"/>
    <points frame="101" keyframe="1" outside="1" occluded="0" points="1000.5,500.0" z_order="0"/>
  </track>
  <track id="1" label="ball" source="manual">
    <points frame="102" keyframe="1" outside="0" occluded="0" points="1010.0,505.0" z_order="0"/>
    <points frame="103" keyframe="0" outside="0" occluded="0" points="1020.0,510.0" z_order="0"/>
    <points frame="104" keyframe="1" outside="0" occluded="0" points="1030.0,515.0" z_order="0"/>
  </track>
  <track id="2" label="ball" source="manual">
    <points frame="900" keyframe="1" outside="0" occluded="0" points="50,60" z_order="0"/>
  </track>
  <track id="3" label="player" source="manual">
    <points frame="100" keyframe="1" outside="0" occluded="0" points="1,1" z_order="0"/>
  </track>
</annotations>
"""


def test_ball_xml_keeps_only_clicked_points(tmp_path):
    p = tmp_path / "rally.ball.xml"
    p.write_text(CVAT_VIDEO_XML)
    lab = L.load_ball_xml(p)
    assert lab.points == {100: (1000.5, 500.0), 102: (1010.0, 505.0), 104: (1030.0, 515.0), 900: (50.0, 60.0)}
    assert lab.dropped_interpolated == 1 and lab.warnings
    assert lab.segments == [(100, 104), (900, 900)]
    assert 103 in lab.frames()  # inside a segment without a point: a "no ball" frame


def test_ball_xml_rejects_two_balls_on_one_frame(tmp_path):
    p = tmp_path / "x.ball.xml"
    p.write_text(CVAT_VIDEO_XML.replace('frame="900"', 'frame="100"'))
    with pytest.raises(ValueError, match="more than one ball"):
        L.load_ball_xml(p)


def test_segments():
    assert L.segments_from_frames([5, 1, 3, 200, 250]) == [(1, 5), (200, 250)]
    assert L.parse_segments("10-20, 30-40") == [(10, 20), (30, 40)]
    with pytest.raises(ValueError):
        L.parse_segments("20-10")


def coco(kpts, bbox=None, file_name="frame_000030.PNG"):
    ann = {"id": 1, "image_id": 7, "keypoints": kpts}
    if bbox is not None:
        ann["bbox"] = bbox
    return {"images": [{"id": 7, "file_name": file_name}], "annotations": [ann]}


def test_pose_coco_reads_frame_number_and_visibility(tmp_path):
    kpts = [0, 0, 0] * 17
    kpts[0:3] = [100, 50, 2]
    kpts[45:48] = [110, 250, 1]
    p = tmp_path / "rally.pose.json"
    p.write_text(json.dumps(coco(kpts)))
    players = L.load_pose_coco(p)[30]
    assert players[0].keypoints[0] == (100.0, 50.0, 2)
    assert players[0].keypoints[15][2] == 1
    assert players[0].bbox == (100.0, 50.0, 10.0, 200.0)  # no box exported: extent of the points
    assert players[0].size == 200.0


def test_pose_coco_rejects_wrong_skeleton(tmp_path):
    p = tmp_path / "x.pose.json"
    p.write_text(json.dumps(coco([1, 2, 2] * 18, bbox=[0, 0, 10, 10])))
    with pytest.raises(ValueError, match="expected 17"):
        L.load_pose_coco(p)


def test_clip_groups(tmp_path):
    assert L.load_clip_groups(tmp_path) == {}
    (tmp_path / "clips.json").write_text(json.dumps({"a": {"height": "low"}, "b": {}}))
    assert L.load_clip_groups(tmp_path) == {"a": "low", "b": "unknown"}
