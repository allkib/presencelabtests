"""FastAPI application factory."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from posedemo.config import Settings
from posedemo.pipeline import PosePipeline, ResultBus
from posedemo.sources.base import FrameSource
from posedemo.vision.detector import PoseDetector
from posedemo.web.routes import router

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    settings: Settings | None = None,
    source: FrameSource | None = None,
    detector: PoseDetector | None = None,
) -> FastAPI:
    """Build the app. ``source``/``detector`` can be injected (tests, other cameras)."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        bus = ResultBus()
        bus.attach_loop(asyncio.get_running_loop())
        src = source
        if src is None:
            from posedemo.sources import build_source

            src = build_source(settings.source, settings.frame_width, settings.frame_height)
        det = detector
        if det is None:
            from posedemo.vision.detector import YoloPoseDetector

            det = YoloPoseDetector(settings)
        pipeline = PosePipeline(settings, src, det, bus)
        app.state.settings = settings
        app.state.bus = bus
        app.state.pipeline = pipeline
        pipeline.start()  # model load + camera open happen on the worker thread
        try:
            yield
        finally:
            await asyncio.to_thread(pipeline.stop)

    app = FastAPI(title="posedemo", lifespan=lifespan)
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
