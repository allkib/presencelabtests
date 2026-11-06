"""Pose rule interface and registry.

To add a pose, subclass ``PoseRule``, set ``name``/``description``/``required``,
implement ``score`` and decorate the class with ``@register``. The classifier,
``/api/poses`` endpoint and web UI pick it up automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, TypeVar

from posedemo.vision.keypoints import KP
from posedemo.vision.types import Person


class PoseRule(ABC):
    name: ClassVar[str]
    description: ClassVar[str] = ""
    required: ClassVar[tuple[KP, ...]] = ()
    # Multiplier applied to the score; use < 1 for "default" poses that should
    # lose ties against more specific ones.
    priority: ClassVar[float] = 1.0
    # Fallback poses (e.g. "neutral") are only chosen when no regular pose passes the
    # threshold, so a generic description never beats a specific one.
    fallback: ClassVar[bool] = False

    def evaluate(self, person: Person, kp_conf: float) -> float:
        """Return a confidence in [0, 1]; 0 if required keypoints aren't visible."""
        if any(person.conf(k) < kp_conf for k in self.required):
            return 0.0
        # Zero out low-confidence keypoints so rules can use ``p.visible(k)``
        # for optional joints (e.g. hips) without knowing the threshold.
        kps = person.keypoints.copy()
        kps[kps[:, 2] < kp_conf, 2] = 0.0
        masked = Person(kps, person.bbox, person.score, person.track_id)
        return max(0.0, min(1.0, self.score(masked))) * self.priority

    @abstractmethod
    def score(self, person: Person) -> float:
        """Score assuming all ``required`` keypoints are visible."""


_REGISTRY: dict[str, PoseRule] = {}
R = TypeVar("R", bound=type[PoseRule])


def register(cls: R) -> R:
    if cls.name in _REGISTRY:
        raise ValueError(f"Pose rule {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls()
    return cls


def all_rules() -> list[PoseRule]:
    return list(_REGISTRY.values())


def get_rule(name: str) -> PoseRule:
    return _REGISTRY[name]
