"""Lightweight per-session IoU tracker.

Ultralytics' built-in tracker keeps its state on the shared model, so it can only
follow one video stream. Browser sessions each get one of these instead, which lets
many clients share a single loaded model while keeping stable person IDs.
"""

from __future__ import annotations

import numpy as np

from posedemo.vision.types import Person


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class IouTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, tuple[np.ndarray, int]] = {}  # id -> (bbox, frames unseen)
        self._next_id = 1

    def update(self, people: list[Person]) -> list[Person]:
        """Assign ``track_id`` to each person (greedy highest-IoU matching)."""
        pairs = sorted(
            (
                (iou(p.bbox, bbox), i, tid)
                for i, p in enumerate(people)
                for tid, (bbox, _) in self._tracks.items()
            ),
            reverse=True,
        )
        matched_people: set[int] = set()
        matched_tracks: set[int] = set()
        for score, i, tid in pairs:
            if score < self.iou_threshold:
                break
            if i in matched_people or tid in matched_tracks:
                continue
            people[i].track_id = tid
            matched_people.add(i)
            matched_tracks.add(tid)

        for i, p in enumerate(people):
            if i not in matched_people:
                p.track_id = self._next_id
                self._next_id += 1

        aged = {
            tid: (bbox, age + 1)
            for tid, (bbox, age) in self._tracks.items()
            if tid not in matched_tracks and age + 1 <= self.max_age
        }
        self._tracks = {**aged, **{p.track_id: (p.bbox.copy(), 0) for p in people}}
        return people
