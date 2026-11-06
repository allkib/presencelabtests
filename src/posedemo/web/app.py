"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from posedemo.config import Settings
from posedemo.pipeline import PosePipeline, ResultBus
from posedemo.poses import PoseClassifier
from posedemo.sources.base import FrameSource
from posedemo.vision.detector import PoseDetector
from posedemo.web.routes import router

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class ModelStatus:
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None

    @property
    def state(self) -> str:
        if self.error:
            return "error"
        return "ready" if self.ready.is_set() else "loading"


def create_app(
    settings: Settings | None = None,
    source: FrameSource | None = None,
    detector: PoseDetector | None = None,
) -> FastAPI:
    """Build the app. ``source``/``detector`` can be injected (tests, other cameras).

    Two input modes share one loaded model:
    * server source (``enable_source``): a camera/file on the server, streamed as MJPEG;
    * browser camera: clients upload frames over ``/ws/infer``.
    """
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        det = detector
        if det is None:
            from posedemo.vision.detector import YoloPoseDetector

            det = YoloPoseDetector(settings)

        bus = ResultBus()
        bus.attach_loop(asyncio.get_running_loop())
        model = ModelStatus()

        app.state.settings = settings
        app.state.bus = bus
        app.state.detector = det
        app.state.model = model
        app.state.classifier = PoseClassifier(
            threshold=settings.pose_threshold, kp_conf=settings.kp_conf
        )
        app.state.browser_sessions = asyncio.Semaphore(settings.max_browser_sessions)
        app.state.pipeline = None

        async def load_model() -> None:
            try:
                await asyncio.to_thread(det.load)
                model.ready.set()
                log.info("Model ready")
            except Exception as e:  # surfaced via /health and the UI
                log.exception("Model failed to load")
                model.error = str(e)

        # Load in the background so the server (and its health endpoint) come up fast.
        loader = asyncio.create_task(load_model())

        pipeline = None
        if settings.enable_source:
            src = source
            if src is None:
                from posedemo.sources import build_source

                src = build_source(settings.source, settings.frame_width, settings.frame_height)
            pipeline = PosePipeline(settings, src, det, bus, classifier=app.state.classifier)
            app.state.pipeline = pipeline
            pipeline.start()  # its thread also calls det.load(), which is idempotent
        else:
            bus.update_state(status="disabled", error=None)
        try:
            yield
        finally:
            loader.cancel()
            if pipeline is not None:
                await asyncio.to_thread(pipeline.stop)

    app = FastAPI(title="posedemo", lifespan=lifespan)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
