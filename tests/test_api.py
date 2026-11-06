import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from posedemo.config import Settings
from posedemo.web.app import create_app
from tests.conftest import FakeDetector, FakeSource


def wait_for(predicate, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("condition not met in time")


def jpeg(w=640, h=480) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.zeros((h, w, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def client(person_factory):
    source = FakeSource()
    detector = FakeDetector([person_factory(arms="out")])
    app = create_app(Settings(smoothing_window=1), source=source, detector=detector)
    with TestClient(app) as c:
        wait_for(lambda: c.get("/api/state").json().get("people"))
        yield c
    assert source.stopped


@pytest.fixture
def browser_client(person_factory):
    detector = FakeDetector([person_factory(arms="up")])
    app = create_app(Settings(enable_source=False, max_browser_sessions=1), detector=detector)
    with TestClient(app) as c:
        wait_for(lambda: c.get("/health").status_code == 200)
        yield c


# ---- server-source mode ----


def test_state_reports_pose(client):
    state = client.get("/api/state").json()
    assert state["status"] == "running"
    person = state["people"][0]
    assert person["pose"] == "t_pose"
    assert person["id"] == 1
    assert len(person["keypoints"]) == 17


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model"] == "ready" and r.json()["server_source"] == "running"


def test_config_reports_server_source(client):
    cfg = client.get("/api/config").json()
    assert cfg["server_source"] is True
    assert len(cfg["skeleton"]) > 10


def test_poses_endpoint(client):
    names = {p["name"] for p in client.get("/api/poses").json()}
    assert {"t_pose", "hands_up", "squat", "standing"} <= names


def test_snapshot_is_jpeg(client):
    r = client.get("/snapshot.jpg")
    assert r.status_code == 200
    assert r.content[:2] == b"\xff\xd8"


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "/static/app.js" in r.text


def test_websocket_pushes_state(client):
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
        assert msg["people"][0]["pose"] == "t_pose"


def test_startup_error_is_reported(person_factory):
    class BrokenSource(FakeSource):
        def start(self):
            raise RuntimeError("no camera")

    app = create_app(Settings(), source=BrokenSource(), detector=FakeDetector([]))
    with TestClient(app) as c:
        state = wait_for(lambda: (s := c.get("/api/state").json())["status"] == "error" and s)
        assert state["error"] == "no camera"
        r = c.get("/health")
        assert r.status_code == 503
        assert r.json()["server_source_error"] == "no camera"


# ---- browser-camera mode ----


def test_browser_only_mode_has_no_server_source(browser_client):
    assert browser_client.get("/api/config").json()["server_source"] is False
    assert browser_client.get("/health").json()["server_source"] is None


def test_infer_returns_pose_for_uploaded_frame(browser_client):
    with browser_client.websocket_connect("/ws/infer") as ws:
        ws.send_bytes(jpeg())
        msg = ws.receive_json()
        assert msg["type"] == "result"
        assert msg["frame_size"] == [640, 480]
        assert msg["people"][0]["pose"] == "hands_up"
        assert msg["people"][0]["id"] == 1
        ws.send_bytes(jpeg())
        assert ws.receive_json()["people"][0]["id"] == 1  # stable across frames


def test_infer_downscales_large_frames(browser_client):
    with browser_client.websocket_connect("/ws/infer") as ws:
        ws.send_bytes(jpeg(2560, 1440))
        assert ws.receive_json()["frame_size"] == [1280, 720]


def test_infer_rejects_bad_input(browser_client):
    with browser_client.websocket_connect("/ws/infer") as ws:
        ws.send_bytes(b"not a jpeg")
        assert ws.receive_json() == {"type": "error", "error": "could not decode image"}
        ws.send_text("hello")
        assert ws.receive_json()["type"] == "error"
        # Session still works afterwards.
        ws.send_bytes(jpeg())
        assert ws.receive_json()["type"] == "result"


def test_infer_session_limit(browser_client):
    from starlette.websockets import WebSocketDisconnect

    with browser_client.websocket_connect("/ws/infer") as first:
        first.send_bytes(jpeg())
        first.receive_json()
        with browser_client.websocket_connect("/ws/infer") as second:
            with pytest.raises(WebSocketDisconnect) as exc:
                second.receive_json()
            assert exc.value.code == 1013


def test_infer_reports_loading_model(person_factory):
    import threading

    gate = threading.Event()

    class SlowDetector(FakeDetector):
        def load(self):
            gate.wait(5)

    app = create_app(Settings(enable_source=False), detector=SlowDetector([]))
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        with c.websocket_connect("/ws/infer") as ws:
            ws.send_bytes(jpeg())
            assert ws.receive_json() == {"type": "status", "status": "loading", "error": None}
        gate.set()
        wait_for(lambda: c.get("/health").status_code == 200)
