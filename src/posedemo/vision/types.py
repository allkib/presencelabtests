"""Plain data containers passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from posedemo.vision.keypoints import KP


@dataclass(slots=True)
class Person:
    """One detected person.

    ``keypoints`` is a (17, 3) array of (x, y, confidence) in image pixel coordinates
    (y grows downward). ``bbox`` is (x1, y1, x2, y2).
    """

    keypoints: np.ndarray
    bbox: np.ndarray
    score: float = 1.0
    track_id: int | None = None

    def xy(self, kp: KP) -> np.ndarray:
        return self.keypoints[kp, :2]

    def conf(self, kp: KP) -> float:
        return float(self.keypoints[kp, 2])

    def visible(self, *kps: KP) -> bool:
        return all(self.keypoints[k, 2] > 0 for k in kps)

    def mirrored(self, width: int) -> Person:
        """Return a copy with x coordinates flipped for a horizontally mirrored image."""
        kps = self.keypoints.copy()
        kps[:, 0] = np.where(kps[:, 2] > 0, width - kps[:, 0], kps[:, 0])
        x1, y1, x2, y2 = self.bbox
        return Person(kps, np.array([width - x2, y1, width - x1, y2]), self.score, self.track_id)


@dataclass(slots=True)
class PoseResult:
    person: Person
    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.person.track_id,
            "pose": self.label,
            "score": round(self.score, 3),
            "scores": {k: round(v, 3) for k, v in self.scores.items()},
            "bbox": [round(float(v), 1) for v in self.person.bbox],
            "keypoints": [
                [round(float(x), 1), round(float(y), 1), round(float(c), 2)]
                for x, y, c in self.person.keypoints
            ],
            "confidence": round(self.person.score, 3),
        }
