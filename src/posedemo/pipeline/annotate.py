"""Drawing helpers for skeletons and pose labels."""

from __future__ import annotations

import zlib

import cv2
import numpy as np

from posedemo.poses.classifier import UNKNOWN
from posedemo.vision.keypoints import SKELETON
from posedemo.vision.objects import DetectedObject
from posedemo.vision.types import PoseResult

_PALETTE = [  # BGR, matching the web UI palette
    (250, 139, 167),
    (231, 140, 123),
    (252, 132, 192),
    (212, 234, 94),
    (252, 171, 240),
    (252, 180, 165),
    (77, 211, 252),
    (212, 168, 249),
]
_UNKNOWN_COLOR = (115, 115, 115)


def color_for(label: str) -> tuple[int, int, int]:
    if label == UNKNOWN:
        return _UNKNOWN_COLOR
    return _PALETTE[zlib.crc32(label.encode()) % len(_PALETTE)]


def pretty(label: str) -> str:
    return label.replace("_", " ").title()


def draw_result(img: np.ndarray, r: PoseResult, kp_conf: float) -> None:
    color = color_for(r.label)
    kps = r.person.keypoints
    thickness = max(2, img.shape[1] // 400)

    for a, b in SKELETON:
        if kps[a, 2] >= kp_conf and kps[b, 2] >= kp_conf:
            pa = tuple(int(v) for v in kps[a, :2])
            pb = tuple(int(v) for v in kps[b, :2])
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    for x, y, c in kps:
        if c >= kp_conf:
            cv2.circle(img, (int(x), int(y)), thickness + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), thickness + 2, color, 1, cv2.LINE_AA)

    x1, y1, x2, y2 = (int(v) for v in r.person.bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    tid = f"#{r.person.track_id} " if r.person.track_id is not None else ""
    text = f"{tid}{pretty(r.label)} {r.score:.2f}"
    if r.holding:
        text += f" + {', '.join(r.holding)}"
    _label(img, text, (max(x1, 0), max(y1, 75)), color)  # keep clear of the HUD


def _label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    fg: tuple[int, int, int] = (20, 20, 20),
) -> None:
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    (w, h), base = cv2.getTextSize(text, font, scale, th)
    x, y = org
    cv2.rectangle(img, (x, y - h - base - 6), (x + w + 10, y), color, -1)
    cv2.putText(img, text, (x + 5, y - base - 2), font, scale, fg, th, cv2.LINE_AA)


def draw_hud(img: np.ndarray, fps: float, people: int) -> None:
    _label(img, f"{fps:4.1f} FPS | {people} people", (10, 34), (40, 40, 40), fg=(240, 240, 240))


OBJECT_COLOR = (245, 245, 245)  # near-white (BGR)


def draw_object(img: np.ndarray, obj: DetectedObject) -> None:
    x1, y1, x2, y2 = (int(v) for v in obj.bbox)
    thickness = max(2, img.shape[1] // 500)
    cv2.rectangle(img, (x1, y1), (x2, y2), OBJECT_COLOR, thickness, cv2.LINE_AA)
    _label(img, f"{obj.label} {obj.score:.2f}", (max(x1, 0), max(y2 + 28, 28)), OBJECT_COLOR)


def annotate(
    img: np.ndarray,
    results: list[PoseResult],
    fps: float,
    kp_conf: float,
    objects: list[DetectedObject] = (),
) -> np.ndarray:
    for obj in objects:
        draw_object(img, obj)
    for r in results:
        draw_result(img, r, kp_conf)
    draw_hud(img, fps, len(results))
    return img
