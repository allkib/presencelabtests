"""Background inference loop: source → detector → classifier → annotate → bus."""

from __future__ import annotations

import logging
import threading
import time

import cv2

from posedemo.config import Settings
from posedemo.pipeline.annotate import annotate
from posedemo.pipeline.bus import ResultBus
from posedemo.poses import LabelSmoother, PoseClassifier
from posedemo.sources.base import FrameSource
from posedemo.vision.association import attach_held_objects
from posedemo.vision.detector import PoseDetector
from posedemo.vision.objects import PeriodicObjects
from posedemo.vision.tracking import IouTracker

log = logging.getLogger(__name__)


class PosePipeline:
    def __init__(
        self,
        settings: Settings,
        source: FrameSource,
        detector: PoseDetector,
        bus: ResultBus,
        classifier: PoseClassifier | None = None,
        objects: PeriodicObjects | None = None,
    ) -> None:
        self.settings = settings
        self.source = source
        self.detector = detector
        self.bus = bus
        self.classifier = classifier or PoseClassifier(
            threshold=settings.pose_threshold, kp_conf=settings.kp_conf
        )
        self.tracker = IouTracker()
        self.objects = objects or PeriodicObjects(None)
        self.smoother = LabelSmoother(window=settings.smoothing_window)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps = 0.0

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="pose-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        try:
            self.source.stop()
        except Exception:  # pragma: no cover
            log.exception("Error stopping source")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            self.bus.update_state(status="loading_model", error=None)
            self.detector.load()
            self.bus.update_state(status="opening_camera")
            self.source.start()
        except Exception as e:
            log.exception("Pipeline failed to start")
            self.bus.update_state(status="error", error=str(e))
            return

        self.bus.update_state(status="running")
        last = time.perf_counter()
        while not self._stop.is_set():
            frame = self.source.read(timeout=1.0)
            if frame is None:
                continue
            try:
                self._process(frame.image, frame.timestamp)
            except Exception as e:  # keep the loop alive on transient errors
                log.exception("Frame processing failed")
                self.bus.update_state(status="running", error=str(e))
            now = time.perf_counter()
            dt = now - last
            last = now
            if dt > 0:
                self._fps = 1.0 / dt if self._fps == 0 else 0.9 * self._fps + 0.1 / dt

    def _process(self, image, timestamp: float) -> None:
        s = self.settings
        people = self.tracker.update(self.detector.detect(image))
        objects = self.objects.update(image)
        results = self.smoother.update(self.classifier.classify(people))
        attach_held_objects(results, objects, s.kp_conf)

        h, w = image.shape[:2]
        if s.mirror:
            image = cv2.flip(image, 1)
            for r in results:
                r.person = r.person.mirrored(w)
            objects = [o.mirrored(w) for o in objects]

        annotate(image, results, self._fps, s.kp_conf, objects)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, s.jpeg_quality])
        state = {
            "status": "running",
            "error": None,
            "timestamp": timestamp,
            "fps": round(self._fps, 1),
            "frame_size": [w, h],
            "people": [r.to_dict() for r in results],
            "objects": [o.to_dict() for o in objects],
        }
        self.bus.publish(buf.tobytes() if ok else None, state)
