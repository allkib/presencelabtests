"""Link detected objects to the person holding them (nearest hand inside the box)."""

from __future__ import annotations

import numpy as np

from posedemo.vision.keypoints import KP
from posedemo.vision.objects import DetectedObject
from posedemo.vision.types import PoseResult

HANDS = ((KP.LEFT_WRIST, KP.LEFT_ELBOW), (KP.RIGHT_WRIST, KP.RIGHT_ELBOW))


def hand_points(r: PoseResult, kp_conf: float) -> list[np.ndarray]:
    """Visible wrists; a side's elbow stands in when its wrist isn't visible."""
    kps = r.person.keypoints
    points = []
    for wrist, elbow in HANDS:
        if kps[wrist, 2] >= kp_conf:
            points.append(kps[wrist, :2])
        elif kps[elbow, 2] >= kp_conf:
            points.append(kps[elbow, :2])
    return points


def _area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def attach_held_objects(
    results: list[PoseResult],
    objects: list[DetectedObject],
    kp_conf: float = 0.35,
    expand: float = 0.25,
    max_relative_size: float = 0.5,
) -> list[PoseResult]:
    """Set ``holding`` on each result: objects whose (expanded) box contains one of the
    person's hands. Each object goes to the person whose hand is closest to its centre.

    Objects larger than ``max_relative_size`` × the person's box (a bus, a sofa, a desk
    the person stands in front of) are never "held"."""
    for r in results:
        r.holding = []
    hands = [(r, pt) for r in results for pt in hand_points(r, kp_conf)]
    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        dx, dy = (x2 - x1) * expand, (y2 - y1) * expand
        centre = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        inside = [
            (float(np.linalg.norm(pt - centre)), r)
            for r, pt in hands
            if x1 - dx <= pt[0] <= x2 + dx
            and y1 - dy <= pt[1] <= y2 + dy
            and _area(obj.bbox) <= max_relative_size * _area(r.person.bbox)
        ]
        if inside:
            owner = min(inside, key=lambda t: t[0])[1]
            if obj.label not in owner.holding:
                owner.holding.append(obj.label)
    return results
