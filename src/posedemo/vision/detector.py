"""Pose detectors. ``PoseDetector`` is the interface; ``YoloPoseDetector`` wraps Ultralytics."""

from __future__ import annotations

import logging
import threading
from typing import Protocol, runtime_checkable

import numpy as np

from posedemo.config import Settings
from posedemo.vision.types import Person

log = logging.getLogger(__name__)


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
        self.device = settings.resolve_device()
        self._model = None
        # One model is shared by the server pipeline and all browser sessions;
        # Ultralytics predictors aren't thread-safe, so inference is serialised.
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is None:
                self._load()

    def _load(self) -> None:
        from ultralytics import YOLO  # heavy import, deferred

        path = self.settings.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Loading %s on device=%s", path, self.device)
        model = YOLO(str(path))
        # Warm up so the first real frame isn't slow.
        model.predict(
            np.zeros((self.settings.imgsz, self.settings.imgsz, 3), dtype=np.uint8),
            device=self.device,
            imgsz=self.settings.imgsz,
            verbose=False,
        )
        self._model = model

    def detect(self, image: np.ndarray) -> list[Person]:
        if self._model is None:
            raise RuntimeError("Detector not loaded; call load() first")
        s = self.settings
        # Plain predict (no Ultralytics tracker): its tracker state lives on the shared
        # model and would mix up IDs between streams. Each stream uses an IouTracker.
        with self._lock:
            result = self._model.predict(
                image, conf=s.conf, imgsz=s.imgsz, device=self.device, verbose=False
            )[0]
        if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
            return []
        kps = result.keypoints.data.cpu().numpy()  # (n, 17, 3)
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [
            Person(keypoints=kps[i], bbox=boxes[i], score=float(scores[i]))
            for i in range(len(boxes))
        ]
