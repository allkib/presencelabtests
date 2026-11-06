"""HTTP + WebSocket endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse

from posedemo.pipeline.bus import ResultBus
from posedemo.poses import all_rules

router = APIRouter()
INDEX = Path(__file__).parent / "static" / "index.html"
BOUNDARY = "frame"


def _bus(request: Request | WebSocket) -> ResultBus:
    return request.app.state.bus


@router.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(INDEX)


@router.get("/health")
async def health(request: Request) -> dict:
    pipeline = request.app.state.pipeline
    state = _bus(request).latest.state
    return {"ok": state.get("status") != "error", "pipeline_alive": pipeline.running, **state}


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
