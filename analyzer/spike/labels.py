"""Read the hand labels exported from CVAT (see LABELING.md).

Ball: CVAT for video 1.1 XML, one `points` shape per frame on the `ball` label.
Pose: COCO Keypoints 1.0 JSON, one `player` skeleton per player.

Both parsers keep only what was actually clicked. An interpolated track frame (track mode) is a
guess made by CVAT, so it is dropped and counted, never used as ground truth.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# Two labeled ball frames further apart than this start a new segment. A rally rarely hides the
# ball for 1.5 s, and the gap between two labeled runs is much longer.
MAX_SEGMENT_GAP = 90


@dataclass
class BallLabels:
    points: dict[int, tuple[float, float]]
    segments: list[tuple[int, int]]  # inclusive frame ranges; frames inside without a point are "no ball"
    dropped_interpolated: int = 0
    warnings: list[str] = field(default_factory=list)

    def frames(self) -> list[int]:
        return [f for a, b in self.segments for f in range(a, b + 1)]


def segments_from_frames(frames: list[int], max_gap: int = MAX_SEGMENT_GAP) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    for f in sorted(set(frames)):
        if segments and f - segments[-1][1] <= max_gap:
            segments[-1] = (segments[-1][0], f)
        else:
            segments.append((f, f))
    return segments


def parse_segments(text: str) -> list[tuple[int, int]]:
    """'1200-1699,4000-4499' -> [(1200, 1699), (4000, 4499)]."""
    out = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        a, _, b = part.partition("-")
        start, stop = int(a), int(b)
        if stop < start:
            raise ValueError(f"segment {part}: end before start")
        out.append((start, stop))
    return out


def load_ball_xml(path: Path, label: str = "ball", segments: list[tuple[int, int]] | None = None) -> BallLabels:
    root = ET.parse(path).getroot()
    points: dict[int, tuple[float, float]] = {}
    warnings: list[str] = []
    dropped = 0

    def add(frame: int, raw: str) -> None:
        coords = [tuple(float(v) for v in p.split(",")) for p in raw.split(";") if p]
        if len(coords) != 1:
            warnings.append(f"frame {frame}: {len(coords)} points in one shape, used the first")
        if frame in points:
            raise ValueError(f"{path}: frame {frame} has more than one ball; label only the ball in play")
        points[frame] = coords[0][:2]

    for track in root.iter("track"):
        if track.get("label") != label:
            continue
        for shape in track.iter("points"):
            if shape.get("outside") == "1":
                continue
            if shape.get("keyframe") != "1":
                dropped += 1  # interpolated by CVAT, not seen by a human
                continue
            add(int(shape.get("frame")), shape.get("points", ""))
    for image in root.iter("image"):  # image-task export: shapes nested per frame
        for shape in image.iter("points"):
            if shape.get("label") == label:
                add(int(image.get("id")), shape.get("points", ""))

    if dropped:
        warnings.append(f"{dropped} interpolated track frames ignored; label in shape mode (LABELING.md)")
    segs = segments if segments is not None else segments_from_frames(list(points))
    return BallLabels(points, segs, dropped, warnings)


@dataclass
class PosePlayer:
    keypoints: list[tuple[float, float, int]]  # COCO visibility: 0 not labeled, 1 occluded, 2 visible
    bbox: tuple[float, float, float, float]  # x, y, w, h

    @property
    def size(self) -> float:
        """Player size used to scale the PCK distance: the longer side of the box."""
        return max(self.bbox[2], self.bbox[3])


def _frame_from_file_name(name: str) -> int:
    digits = re.findall(r"\d+", Path(name).stem)
    if not digits:
        raise ValueError(f"cannot read a frame number from image name {name!r}")
    return int(digits[-1])


def load_pose_coco(path: Path) -> dict[int, list[PosePlayer]]:
    data = json.loads(path.read_text())
    frame_of = {img["id"]: _frame_from_file_name(img["file_name"]) for img in data["images"]}
    out: dict[int, list[PosePlayer]] = {}
    for ann in data["annotations"]:
        raw = ann.get("keypoints") or []
        if len(raw) != 17 * 3:
            raise ValueError(f"{path}: annotation {ann.get('id')} has {len(raw) // 3} keypoints, expected 17")
        kpts = [(float(raw[i]), float(raw[i + 1]), int(raw[i + 2])) for i in range(0, len(raw), 3)]
        labeled = [(x, y) for x, y, v in kpts if v > 0]
        if not labeled:
            continue
        bbox = tuple(float(v) for v in ann.get("bbox") or (0, 0, 0, 0))
        if bbox[2] * bbox[3] <= 0:  # no box exported: use the extent of the labeled points
            xs, ys = [p[0] for p in labeled], [p[1] for p in labeled]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        out.setdefault(frame_of[ann["image_id"]], []).append(PosePlayer(kpts, bbox))
    return out


def load_clip_groups(labels_dir: Path) -> dict[str, str]:
    """Optional labels/clips.json: {"<clip>": {"height": "low"}}. Results are reported per height."""
    path = labels_dir / "clips.json"
    if not path.is_file():
        return {}
    return {clip: str(info.get("height", "unknown")) for clip, info in json.loads(path.read_text()).items()}
