"""HTTP + WebSocket endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from posedemo.pipeline.bus import ResultBus
from posedemo.poses import all_rules
from posedemo.vision.keypoints import SKELETON
from posedemo.web.sessions import BrowserSession, FrameDecodeError

router = APIRouter()
INDEX = Path(__file__).parent / "static" / "index.html"
BOUNDARY = "frame"


def _bus(request: Request | WebSocket) -> ResultBus:
    return request.app.state.bus


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX)


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """200 when the model is loaded and the server source (if enabled) is healthy, else 503."""
    model = request.app.state.model
    pipeline = request.app.state.pipeline
    source = _bus(request).latest.state
    source_ok = pipeline is None or (pipeline.running and source.get("status") != "error")
    ok = model.state == "ready" and source_ok
    body = {
        "ok": ok,
        "model": model.state,
        "model_error": model.error,
        "server_source": None if pipeline is None else source.get("status"),
        "server_source_error": None if pipeline is None else source.get("error"),
    }
    return JSONResponse(body, status_code=200 if ok else 503)


@router.get("/api/config")
async def config(request: Request) -> dict:
    """What the UI needs to know: available modes and how to draw skeletons."""
    s = request.app.state.settings
    return {
        "server_source": request.app.state.pipeline is not None,
        "model": request.app.state.model.state,
        "max_frame_side": s.max_frame_side,
        "kp_conf": s.kp_conf,
        "skeleton": [[int(a), int(b)] for a, b in SKELETON],
    }


@router.get("/api/state")
async def state(request: Request) -> dict:
    return _bus(request).latest.state


@router.get("/api/poses")
async def poses() -> list[dict]:
    return [{"name": r.name, "description": r.description} for r in all_rules()]


@router.get("/snapshot.jpg")
async def snapshot(request: Request) -> Response:
    jpeg = _bus(request).latest.jpeg
    if jpeg is None:
        return Response(status_code=503, content="no frame yet")
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/stream")
async def stream(request: Request) -> StreamingResponse:
    """MJPEG stream of annotated frames (use as an <img> src)."""
    bus = _bus(request)

    async def frames():
        seq = 0
        while not await request.is_disconnected():
            snap = await bus.wait_next(seq, timeout=2.0)
            if snap.seq == seq or snap.jpeg is None:
                continue
            seq = snap.seq
            yield (
                (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(snap.jpeg)}\r\n\r\n"
                ).encode()
                + snap.jpeg
                + b"\r\n"
            )

    return StreamingResponse(frames(), media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}")


@router.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Pushes the pose state JSON every time a new frame is processed."""
    await websocket.accept()
    bus = _bus(websocket)
    seq = -1
    try:
        while True:
            snap = await bus.wait_next(seq, timeout=5.0)
            if snap.seq == seq:
                continue
            seq = snap.seq
            await websocket.send_json(snap.state)
    except WebSocketDisconnect:
        pass


@router.websocket("/ws/infer")
async def ws_infer(websocket: WebSocket) -> None:
    """Browser-camera mode: client sends JPEG frames (binary), server replies with JSON.

    The client should send its next frame only after receiving a reply; this gives
    natural back-pressure so slow servers never build up a queue.
    """
    await websocket.accept()
    app = websocket.app
    settings = app.state.settings
    slots: asyncio.Semaphore = app.state.browser_sessions
    if slots.locked():
        await websocket.close(code=1013, reason="Server busy, try again shortly")
        return

    async with slots:
        session = BrowserSession(settings, app.state.detector, app.state.classifier)
        try:
            while True:
                msg = await websocket.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if not data:
                    await websocket.send_json({"type": "error", "error": "send JPEG frames"})
                    continue
                if len(data) > settings.max_upload_bytes:
                    await websocket.send_json({"type": "error", "error": "frame too large"})
                    continue
                model = app.state.model
                if model.state != "ready":
                    await websocket.send_json(
                        {"type": "status", "status": model.state, "error": model.error}
                    )
                    continue
                try:
                    result = await asyncio.to_thread(session.process, data)
                except FrameDecodeError as e:
                    await websocket.send_json({"type": "error", "error": str(e)})
                    continue
                await websocket.send_json(result)
        except WebSocketDisconnect:
            pass
