"""Threaded OpenCV webcam reader that always serves the most recent frame."""

from __future__ import annotations

import logging
import threading
import time

import cv2

from posedemo.sources.base import Frame, SourceError

log = logging.getLogger(__name__)


class WebcamSource:
    """Grabs frames on a background thread, keeping only the latest one.

    Dropping stale frames keeps latency low when inference is slower than the camera.
    Frames are returned un-mirrored so left/right keypoints stay anatomically correct;
    mirroring for display happens in the pipeline.
    """

    def __init__(
        self,
        index: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._last_returned = -1
        self._counter = 0

    def start(self) -> None:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            raise SourceError(
                f"Could not open camera index {self.index}. On macOS, grant your terminal "
                "Camera access in System Settings → Privacy & Security → Camera."
            )
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="webcam", daemon=True)
        self._thread.start()
        log.info("Camera %s opened", self.index)

    def _loop(self) -> None:
        assert self._cap is not None
        failures = 0
        while self._running.is_set():
            ok, img = self._cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    log.error("Camera stopped delivering frames")
                    break
                time.sleep(0.02)
                continue
            failures = 0
            with self._cond:
                self._counter += 1
                self._latest = Frame(img, time.time(), self._counter)
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> Frame | None:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._latest is not None and self._latest.index != self._last_returned,
                timeout=timeout,
            )
            if not ok or self._latest is None:
                return None
            self._last_returned = self._latest.index
            return self._latest

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
