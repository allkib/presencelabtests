import time

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


@pytest.fixture
def client(person_factory):
    source = FakeSource()
    detector = FakeDetector([person_factory(arms="out", track_id=3)])
    app = create_app(Settings(smoothing_window=1), source=source, detector=detector)
    with TestClient(app) as c:
        wait_for(lambda: c.get("/api/state").json().get("people"))
        yield c
    assert source.stopped


def test_state_reports_pose(client):
    state = client.get("/api/state").json()
    assert state["status"] == "running"
    assert state["people"][0]["pose"] == "t_pose"
    assert state["people"][0]["id"] == 1


def test_health(client):
    body = client.get("/health").json()
    assert body["ok"] is True and body["pipeline_alive"] is True


def test_poses_endpoint(client):
    names = {p["name"] for p in client.get("/api/poses").json()}
    assert {"t_pose", "hands_up", "squat", "standing"} <= names


def test_snapshot_is_jpeg(client):
    r = client.get("/snapshot.jpg")
    assert r.status_code == 200
    assert r.content[:2] == b"\xff\xd8"


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "/stream" in r.text


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
        assert c.get("/health").json()["ok"] is False
