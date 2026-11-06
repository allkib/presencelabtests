"""General object detection (COCO-80 classes such as cell phone, cup, bottle, book)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from posedemo.config import Settings
from posedemo.vision.yolo import YoloModel


@dataclass(slots=True)
class DetectedObject:
    label: str
    score: float
    bbox: np.ndarray  # x1, y1, x2, y2 in image pixels

    def mirrored(self, width: int) -> DetectedObject:
        x1, y1, x2, y2 = self.bbox
        return DetectedObject(self.label, self.score, np.array([width - x2, y1, width - x1, y2]))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 3),
            "bbox": [round(float(v), 1) for v in self.bbox],
        }


@runtime_checkable
class ObjectDetector(Protocol):
    def load(self) -> None: ...

    def detect(self, image: np.ndarray) -> list[DetectedObject]: ...


class YoloObjectDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._yolo = YoloModel(settings, settings.object_model)
        self._class_ids: list[int] | None = None

    def load(self) -> None:
        self._yolo.load()
        names = self._yolo.names
        wanted = self.settings.object_classes
        if wanted is None:
            self._class_ids = [i for i, n in names.items() if n != "person"]
        else:
            unknown = set(wanted) - set(names.values())
            if unknown:
                raise ValueError(f"Unknown object classes: {sorted(unknown)}")
            self._class_ids = [i for i, n in names.items() if n in wanted]

    def detect(self, image: np.ndarray) -> list[DetectedObject]:
        result = self._yolo.predict(image, conf=self.settings.object_conf, classes=self._class_ids)
        if result.boxes is None or len(result.boxes) == 0:
            return []
        names = result.names
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.int().cpu().tolist()
        return [DetectedObject(names[c], float(scores[i]), boxes[i]) for i, c in enumerate(classes)]


class PeriodicObjects:
    """Per-stream helper: runs object detection every ``every_n`` frames and reuses the
    last result in between. Returns nothing until ``ready()`` is true (model loading)."""

    def __init__(
        self,
        detector: ObjectDetector | None,
        every_n: int = 1,
        ready: Callable[[], bool] = lambda: True,
    ) -> None:
        self.detector = detector
        self.every_n = max(1, every_n)
        self.ready = ready
        self._frame = 0
        self._last: list[DetectedObject] = []

    def update(self, image: np.ndarray) -> list[DetectedObject]:
        if self.detector is None or not self.ready():
            return []
        if self._frame % self.every_n == 0:
            self._last = self.detector.detect(image)
        self._frame += 1
        return self._last
