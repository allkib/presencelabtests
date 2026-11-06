"""Video file / image / stream-URL source, paced to the file's FPS. Loops by default."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from posedemo.sources.base import Frame, SourceError

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VideoFileSource:
    def __init__(self, path: str, loop: bool = True, fps: float | None = None) -> None:
        self.path = path
        self.loop = loop
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._image = None
        self._counter = 0
        self._next_t = 0.0

    def start(self) -> None:
        if Path(self.path).suffix.lower() in IMAGE_EXTS:
            self._image = cv2.imread(self.path)
            if self._image is None:
                raise SourceError(f"Could not read image {self.path}")
            self.fps = self.fps or 15.0
            return
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise SourceError(f"Could not open video source {self.path}")
        self._cap = cap
        self.fps = self.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self, timeout: float = 1.0) -> Frame | None:
        delay = self._next_t - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, timeout))
        self._next_t = time.monotonic() + 1.0 / (self.fps or 30.0)

        if self._image is not None:
            img = self._image.copy()
        else:
            assert self._cap is not None
            ok, img = self._cap.read()
            if not ok and self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = self._cap.read()
            if not ok:
                return None
        self._counter += 1
        return Frame(img, time.time(), self._counter)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
