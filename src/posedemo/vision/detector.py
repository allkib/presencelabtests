"""Pose detectors. ``PoseDetector`` is the interface; ``YoloPoseDetector`` wraps Ultralytics."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

from posedemo.config import Settings
from posedemo.vision.types import Person

log = logging.getLogger(__name__)


@runtime_checkable
class PoseDetector(Protocol):
    def load(self) -> None:
        """Load weights / warm up. Called once from the pipeline thread."""
        ...

    def detect(self, image: np.ndarray) -> list[Person]: ...


class YoloPoseDetector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.resolve_device()
        self._model = None

    def load(self) -> None:
        from ultralytics import YOLO  # heavy import, deferred

        path = self.settings.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Loading %s on device=%s", path, self.device)
        self._model = YOLO(str(path))
        # Warm up so the first real frame isn't slow.
        self._model.predict(
            np.zeros((self.settings.imgsz, self.settings.imgsz, 3), dtype=np.uint8),
            device=self.device,
            imgsz=self.settings.imgsz,
            verbose=False,
        )

    def detect(self, image: np.ndarray) -> list[Person]:
        if self._model is None:
            raise RuntimeError("Detector not loaded; call load() first")
        s = self.settings
        result = self._model.track(
            image,
            persist=True,
            tracker=s.tracker,
            conf=s.conf,
            imgsz=s.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []
        kps = result.keypoints.data.cpu().numpy()  # (n, 17, 3)
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id
        ids = ids.int().cpu().tolist() if ids is not None else [None] * len(boxes)
        return [
            Person(keypoints=kps[i], bbox=boxes[i], score=float(scores[i]), track_id=ids[i])
            for i in range(len(boxes))
        ]
