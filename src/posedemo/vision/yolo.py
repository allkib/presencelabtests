"""Shared loader for Ultralytics models: lazy, idempotent, thread-safe."""

from __future__ import annotations

import logging
import threading

import numpy as np

from posedemo.config import Settings

log = logging.getLogger(__name__)


class YoloModel:
    """One loaded YOLO model shared by the server pipeline and every browser session.

    Ultralytics predictors aren't thread-safe, so inference is serialised per model.
    """

    def __init__(self, settings: Settings, weights: str) -> None:
        self.settings = settings
        self.weights = weights
        self.device = settings.resolve_device()
        self._model = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def names(self) -> dict[int, str]:
        return self._model.names if self._model is not None else {}

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from ultralytics import YOLO  # heavy import, deferred

            path = self.settings.weights_path(self.weights)
            path.parent.mkdir(parents=True, exist_ok=True)
            log.info("Loading %s on device=%s", path, self.device)
            model = YOLO(str(path))
            # Warm up so the first real frame isn't slow.
            imgsz = self.settings.imgsz
            model.predict(
                np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                device=self.device,
                imgsz=imgsz,
                verbose=False,
            )
            self._model = model

    def predict(self, image: np.ndarray, **kwargs):
        """Run inference on one image and return the Ultralytics ``Results``."""
        if self._model is None:
            raise RuntimeError(f"{self.weights} not loaded; call load() first")
        with self._lock:
            return self._model.predict(
                image, imgsz=self.settings.imgsz, device=self.device, verbose=False, **kwargs
            )[0]
