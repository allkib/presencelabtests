"""Shared fixtures: synthetic skeletons and fake source/detector (no camera, no model)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from posedemo.sources.base import Frame
from posedemo.vision.keypoints import KP
from posedemo.vision.types import Person

# A person facing the camera: their LEFT side appears at larger x. y grows downward.
BASE = {
    KP.NOSE: (300, 100), KP.LEFT_EYE: (310, 92), KP.RIGHT_EYE: (290, 92),
    KP.LEFT_EAR: (320, 98), KP.RIGHT_EAR: (280, 98),
    KP.LEFT_SHOULDER: (340, 150), KP.RIGHT_SHOULDER: (260, 150),
    KP.LEFT_HIP: (330, 300), KP.RIGHT_HIP: (270, 300),
    KP.LEFT_KNEE: (330, 420), KP.RIGHT_KNEE: (270, 420),
    KP.LEFT_ANKLE: (330, 540), KP.RIGHT_ANKLE: (270, 540),
}  # fmt: skip
ARMS = {
    "down": {KP.LEFT_ELBOW: (345, 225), KP.LEFT_WRIST: (350, 300),
             KP.RIGHT_ELBOW: (255, 225), KP.RIGHT_WRIST: (250, 300)},
    "out": {KP.LEFT_ELBOW: (415, 150), KP.LEFT_WRIST: (490, 152),
            KP.RIGHT_ELBOW: (185, 150), KP.RIGHT_WRIST: (110, 152)},
    "up": {KP.LEFT_ELBOW: (345, 75), KP.LEFT_WRIST: (350, 0),
           KP.RIGHT_ELBOW: (255, 75), KP.RIGHT_WRIST: (250, 0)},
    "left_up": {KP.LEFT_ELBOW: (345, 75), KP.LEFT_WRIST: (350, 0),
                KP.RIGHT_ELBOW: (255, 225), KP.RIGHT_WRIST: (250, 300)},
    "right_up": {KP.LEFT_ELBOW: (345, 225), KP.LEFT_WRIST: (350, 300),
                 KP.RIGHT_ELBOW: (255, 75), KP.RIGHT_WRIST: (250, 0)},
    "crossed": {KP.LEFT_ELBOW: (335, 235), KP.LEFT_WRIST: (275, 200),
                KP.RIGHT_ELBOW: (265, 235), KP.RIGHT_WRIST: (325, 200)},
}  # fmt: skip
SQUAT_LEGS = {
    KP.LEFT_HIP: (330, 400), KP.RIGHT_HIP: (270, 400),
    KP.LEFT_KNEE: (380, 420), KP.RIGHT_KNEE: (220, 420),
    KP.LEFT_ANKLE: (330, 540), KP.RIGHT_ANKLE: (270, 540),
}  # fmt: skip


def make_person(arms: str = "down", squat: bool = False, track_id: int | None = 1, **override):
    pts = {**BASE, **ARMS[arms]}
    if squat:
        pts.update(SQUAT_LEGS)
    pts.update(override)
    kps = np.zeros((17, 3), dtype=float)
    for k, (x, y) in pts.items():
        kps[k] = (x, y, 0.95)
    xs, ys = kps[:, 0], kps[:, 1]
    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
    return Person(keypoints=kps, bbox=bbox, score=0.9, track_id=track_id)


@pytest.fixture
def person_factory():
    return make_person


class FakeSource:
    def __init__(self, shape=(480, 640, 3)):
        self.shape = shape
        self.i = 0
        self.started = self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self, timeout: float = 1.0) -> Frame | None:
        time.sleep(0.01)
        self.i += 1
        return Frame(np.zeros(self.shape, dtype=np.uint8), time.time(), self.i)

    def stop(self) -> None:
        self.stopped = True


class FakeDetector:
    def __init__(self, people: list[Person]):
        self.people = people
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def detect(self, image) -> list[Person]:
        return [Person(p.keypoints.copy(), p.bbox.copy(), p.score, p.track_id) for p in self.people]
