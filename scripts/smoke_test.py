"""End-to-end smoke test for a running posedemo server (local, Docker or Dokploy).

    python scripts/smoke_test.py                          # http://127.0.0.1:8000
    python scripts/smoke_test.py https://pose.example.com
    python scripts/smoke_test.py URL path/to/photo.jpg

Checks /health and /api/config, then sends a photo of people through the browser-camera
WebSocket (/ws/infer) and verifies that people and 17 keypoints come back. Needs only
the stdlib plus `websockets` (bundled with uvicorn[standard]); defaults to Ultralytics'
sample bus.jpg when ultralytics is installed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path


def default_image() -> Path:
    from ultralytics.utils import ASSETS

    return Path(ASSETS) / "bus.jpg"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.load(r)


async def infer(ws_url: str, data: bytes, frames: int = 5) -> dict:
    import websockets

    async with websockets.connect(ws_url, max_size=2**22) as ws:
        msg: dict = {}
        start = time.perf_counter()
        for _ in range(frames):
            await ws.send(data)
            msg = json.loads(await ws.recv())
            if msg.get("type") != "result":
                raise AssertionError(f"unexpected reply: {msg}")
        msg["_fps"] = frames / (time.perf_counter() - start)
        return msg


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    image = Path(sys.argv[2]) if len(sys.argv) > 2 else default_image()
    ws_base = "wss" + base[5:] if base.startswith("https") else "ws" + base[4:]

    health = get_json(f"{base}/health")
    print("health:", health)
    assert health["ok"], "server not healthy"
    # The object model loads after the pose model; give it a moment.
    deadline = time.time() + 60
    while health.get("objects") == "loading" and time.time() < deadline:
        time.sleep(1)
        health = get_json(f"{base}/health")
    print("objects model:", health.get("objects"))

    config = get_json(f"{base}/api/config")
    assert len(config["skeleton"]) > 10, config

    result = asyncio.run(infer(f"{ws_base}/ws/infer", image.read_bytes()))
    people = result["people"]
    fps, size = result["_fps"], result["frame_size"]
    print(f"infer: {len(people)} people, {fps:.1f} frames/s, frame {size}")
    for p in people:
        print(f"  #{p['id']} {p['pose']} ({p['score']})")
    assert people, "no people detected in test image"
    assert all(len(p["keypoints"]) == 17 for p in people)
    objects = [o["label"] for o in result.get("objects", [])]
    print("objects:", objects)
    if health.get("objects") == "ready" and len(sys.argv) <= 2:
        assert "bus" in objects, "object detector didn't find the bus in bus.jpg"
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
