"""Runs every registered pose rule against each detected person."""

from __future__ import annotations

from collections.abc import Iterable

from posedemo.poses.base import PoseRule, all_rules
from posedemo.vision.types import Person, PoseResult

UNKNOWN = "unknown"


class PoseClassifier:
    def __init__(
        self,
        rules: Iterable[PoseRule] | None = None,
        threshold: float = 0.6,
        kp_conf: float = 0.5,
    ) -> None:
        self.rules = list(rules) if rules is not None else all_rules()
        self.threshold = threshold
        self.kp_conf = kp_conf

    def classify_one(self, person: Person) -> PoseResult:
        """Best specific pose above the threshold; otherwise the best fallback pose
        (e.g. "neutral"); otherwise "unknown"."""
        scores = {r.name: r.evaluate(person, self.kp_conf) for r in self.rules}
        for group in (
            [r for r in self.rules if not r.fallback],
            [r for r in self.rules if r.fallback],
        ):
            best = max((r.name for r in group), key=scores.get, default=None)
            if best is not None and scores[best] >= self.threshold:
                return PoseResult(person, best, scores[best], scores)
        top = max(scores.values(), default=0.0)
        return PoseResult(person, UNKNOWN, top, scores)

    def classify(self, people: Iterable[Person]) -> list[PoseResult]:
        return [self.classify_one(p) for p in people]
