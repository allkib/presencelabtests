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
        scores = {r.name: r.evaluate(person, self.kp_conf) for r in self.rules}
        best = max(scores, key=scores.get, default=None)
        if best is None or scores[best] < self.threshold:
            return PoseResult(person, UNKNOWN, scores.get(best, 0.0) if best else 0.0, scores)
        return PoseResult(person, best, scores[best], scores)

    def classify(self, people: Iterable[Person]) -> list[PoseResult]:
        return [self.classify_one(p) for p in people]
