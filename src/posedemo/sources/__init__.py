from posedemo.sources.base import Frame, FrameSource, SourceError
from posedemo.sources.video import VideoFileSource
from posedemo.sources.webcam import WebcamSource


def build_source(spec: str | int, width: int | None = None, height: int | None = None):
    """Create a source from a camera index (``0``) or a file path / stream URL."""
    if isinstance(spec, int) or str(spec).isdigit():
        return WebcamSource(int(spec), width, height)
    return VideoFileSource(str(spec))


__all__ = [
    "Frame",
    "FrameSource",
    "SourceError",
    "VideoFileSource",
    "WebcamSource",
    "build_source",
]
