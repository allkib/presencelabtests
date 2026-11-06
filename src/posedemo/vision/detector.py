"""Pose detectors. ``PoseDetector`` is the interface; ``YoloPoseDetector`` wraps Ultralytics."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from posedemo.config import Settings
from posedemo.vision.types import Person
from posedemo.vision.yolo import YoloModel


@runtime_checkable
class PoseDetector(Protocol):
    def load(self) -> None:
        """Load weights / warm up. Must be idempotent and thread-safe."""
        ...

    def detect(self, image: np.ndarray) -> list[Person]:
        """Detect people (untracked; use ``IouTracker`` per stream for IDs)."""
        ...


class YoloPoseDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._yolo = YoloModel(settings, settings.model)
        self.device = self._yolo.device

    @property
    def loaded(self) -> bool:
        return self._yolo.loaded

    def load(self) -> None:
        self._yolo.load()

    def detect(self, image: np.ndarray) -> list[Person]:
        # Plain predict (no Ultralytics tracker): its tracker state lives on the shared
        # model and would mix up IDs between streams. Each stream uses an IouTracker.
        result = self._yolo.predict(image, conf=self.settings.conf)
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []
        kps = result.keypoints.data.cpu().numpy()  # (n, 17, 3)
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [
            Person(keypoints=kps[i], bbox=boxes[i], score=float(scores[i]))
            for i in range(len(boxes))
        ]
