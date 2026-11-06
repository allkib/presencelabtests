"""Temporal smoothing of pose labels per tracked person to suppress flicker."""

from __future__ import annotations

from collections import Counter, deque

from posedemo.vision.types import PoseResult


class LabelSmoother:
    def __init__(self, window: int = 7, max_missing: int = 30) -> None:
        self.window = max(1, window)
        self.max_missing = max_missing
        self._history: dict[int, deque[str]] = {}
        self._missing: dict[int, int] = {}

    def update(self, results: list[PoseResult]) -> list[PoseResult]:
        seen: set[int] = set()
        for r in results:
            tid = r.person.track_id
            if tid is None or self.window == 1:
                continue
            seen.add(tid)
            hist = self._history.setdefault(tid, deque(maxlen=self.window))
            hist.append(r.label)
            label, _ = Counter(hist).most_common(1)[0]
            if label != r.label:
                r.label = label
                r.score = r.scores.get(label, 0.0)
        self._prune(seen)
        return results

    def _prune(self, seen: set[int]) -> None:
        for tid in list(self._history):
            if tid in seen:
                self._missing[tid] = 0
                continue
            self._missing[tid] = self._missing.get(tid, 0) + 1
            if self._missing[tid] > self.max_missing:
                del self._history[tid]
                del self._missing[tid]

    def reset(self) -> None:
        self._history.clear()
        self._missing.clear()
