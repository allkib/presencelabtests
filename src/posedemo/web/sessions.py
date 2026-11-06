"""Per-client inference for browser-camera mode (``/ws/infer``).

The browser captures its own webcam and uploads JPEG frames. Every session shares
the one loaded model, but has its own tracker and smoother so IDs and labels from
different users never mix.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from posedemo.config import Settings
from posedemo.poses import LabelSmoother, PoseClassifier
from posedemo.vision.association import attach_held_objects
from posedemo.vision.detector import PoseDetector
from posedemo.vision.objects import PeriodicObjects
from posedemo.vision.tracking import IouTracker


class FrameDecodeError(ValueError):
    pass


def decode_jpeg(data: bytes, max_side: int) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FrameDecodeError("could not decode image")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


class BrowserSession:
    def __init__(
        self,
        settings: Settings,
        detector: PoseDetector,
        classifier: PoseClassifier,
        objects: PeriodicObjects | None = None,
    ) -> None:
        self.settings = settings
        self.detector = detector
        self.classifier = classifier
        self.tracker = IouTracker()
        self.objects = objects or PeriodicObjects(None)
        self.smoother = LabelSmoother(window=settings.smoothing_window)

    def process(self, data: bytes) -> dict:
        """Run one uploaded frame through the pipeline (blocking; call off the event loop)."""
        start = time.perf_counter()
        img = decode_jpeg(data, self.settings.max_frame_side)
        people = self.tracker.update(self.detector.detect(img))
        objects = self.objects.update(img)
        results = self.smoother.update(self.classifier.classify(people))
        attach_held_objects(results, objects, self.settings.kp_conf)
        h, w = img.shape[:2]
        return {
            "type": "result",
            "frame_size": [w, h],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            "people": [r.to_dict() for r in results],
            "objects": [o.to_dict() for o in objects],
        }
