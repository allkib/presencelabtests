"""Application settings, loaded from env vars (prefix ``POSEDEMO_``) or a ``.env`` file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSEDEMO_", env_file=".env", extra="ignore")

    # Model / inference
    model: str = "yolo11n-pose.pt"
    models_dir: Path = Path("models")
    device: str = "auto"  # auto | mps | cuda | cpu
    imgsz: int = 640
    conf: float = 0.5  # person detection confidence
    kp_conf: float = 0.35  # keypoint visibility threshold (edge-of-frame joints score ~0.4)

    # Server-side video source (a camera attached to the server, a file or a stream URL).
    # Disable in hosted deployments, where clients stream from their browser instead.
    enable_source: bool = True
    source: str = "0"  # camera index ("0", "1", ...) or video/image path or stream URL
    frame_width: int | None = 1280
    frame_height: int | None = 720
    mirror: bool = True  # flip horizontally so it feels like a mirror

    # Pose classification
    pose_threshold: float = 0.6
    smoothing_window: int = 7

    # Output / web
    jpeg_quality: int = 80
    host: str = "127.0.0.1"
    port: int = 8000
    forwarded_allow_ips: str = "127.0.0.1"  # set "*" behind a reverse proxy (Traefik)

    # Browser-camera sessions (/ws/infer)
    max_browser_sessions: int = 8
    max_upload_bytes: int = 2_000_000
    max_frame_side: int = 1280  # larger uploaded frames are downscaled

    @property
    def model_path(self) -> Path:
        """Weights path inside ``models_dir`` (Ultralytics downloads it there if missing)."""
        p = Path(self.model)
        return p if p.is_absolute() or p.parent != Path(".") else self.models_dir / p

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - torch import edge cases
            pass
        return "cpu"
