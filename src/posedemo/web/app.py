"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from posedemo import __version__
from posedemo.config import Settings
from posedemo.pipeline import PosePipeline, ResultBus
from posedemo.poses import PoseClassifier
from posedemo.sources.base import FrameSource
from posedemo.vision.detector import PoseDetector
from posedemo.vision.objects import ObjectDetector, PeriodicObjects
from posedemo.web.routes import router

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class ModelStatus:
    """Pose model readiness (required) and object model readiness (optional extra)."""

    ready: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None
    objects: str = "disabled"  # disabled | loading | ready | error
    objects_error: str | None = None

    @property
    def state(self) -> str:
        if self.error:
            return "error"
        return "ready" if self.ready.is_set() else "loading"


def create_app(
    settings: Settings | None = None,
    source: FrameSource | None = None,
    detector: PoseDetector | None = None,
    object_detector: ObjectDetector | None = None,
) -> FastAPI:
    """Build the app. ``source``/``detector``/``object_detector`` can be injected (tests,
    other cameras). An object detector is created automatically when ``enable_objects``.

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
        obj_det = object_detector
        if obj_det is None and settings.enable_objects:
            from posedemo.vision.objects import YoloObjectDetector

            obj_det = YoloObjectDetector(settings)

        bus = ResultBus()
        bus.attach_loop(asyncio.get_running_loop())
        model = ModelStatus(objects="loading" if obj_det is not None else "disabled")

        app.state.version = __version__
        app.state.started_at = time.time()
        app.state.settings = settings
        app.state.bus = bus
        app.state.detector = det
        app.state.object_detector = obj_det
        app.state.model = model
        app.state.classifier = PoseClassifier(
            threshold=settings.pose_threshold, kp_conf=settings.kp_conf
        )
        app.state.browser_sessions = asyncio.Semaphore(settings.max_browser_sessions)
        app.state.pipeline = None

        def objects_for_stream() -> PeriodicObjects:
            """Per-stream object detection; yields nothing until its model is ready."""
            return PeriodicObjects(
                obj_det, settings.object_every_n, ready=lambda: model.objects == "ready"
            )

        app.state.objects_for_stream = objects_for_stream

        async def load_model() -> None:
            try:
                await asyncio.to_thread(det.load)
                model.ready.set()
                log.info("Pose model ready")
            except Exception as e:  # surfaced via /health and the UI
                log.exception("Pose model failed to load")
                model.error = str(e)
                return
            if obj_det is None:
                return
            try:  # optional: poses keep working if this fails
                await asyncio.to_thread(obj_det.load)
                model.objects = "ready"
                log.info("Object model ready")
            except Exception as e:
                log.exception("Object model failed to load")
                model.objects, model.objects_error = "error", str(e)

        # Load in the background so the server (and its health endpoint) come up fast.
        loader = asyncio.create_task(load_model())

        pipeline = None
        if settings.enable_source:
            src = source
            if src is None:
                from posedemo.sources import build_source

                src = build_source(settings.source, settings.frame_width, settings.frame_height)
            pipeline = PosePipeline(
                settings,
                src,
                det,
                bus,
                classifier=app.state.classifier,
                objects=objects_for_stream(),
            )
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
