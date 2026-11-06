"""Frame source abstraction so webcams, video files, RTSP, etc. are interchangeable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


class SourceError(RuntimeError):
    """Raised when a frame source cannot be opened or fails permanently."""


@dataclass(frozen=True, slots=True)
class Frame:
    image: np.ndarray  # BGR, HxWx3
    timestamp: float
    index: int


@runtime_checkable
class FrameSource(Protocol):
    def start(self) -> None: ...

    def read(self, timeout: float = 1.0) -> Frame | None:
        """Return the newest frame not yet returned, or ``None`` on timeout."""
        ...

    def stop(self) -> None: ...
