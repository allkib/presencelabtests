import numpy as np

from posedemo.vision.association import attach_held_objects
from posedemo.vision.keypoints import KP
from posedemo.vision.objects import DetectedObject, PeriodicObjects
from posedemo.vision.types import PoseResult
from tests.conftest import FakeObjectDetector, phone_near


def result(person):
    return PoseResult(person, "neutral", 0.7)


def test_phone_in_hand_attaches_to_that_person(person_factory):
    alice = result(person_factory(arms="left_up", track_id=1))
    bob = result(person_factory(arms="down", track_id=2))
    bob.person.keypoints[:, 0] += 600  # stand Bob to the right
    phone = phone_near(alice.person, KP.LEFT_WRIST)
    attach_held_objects([alice, bob], [phone])
    assert alice.holding == ["cell phone"]
    assert bob.holding == []


def test_far_object_attaches_to_nobody(person_factory):
    r = result(person_factory())
    far = DetectedObject("cup", 0.9, np.array([2000, 2000, 2050, 2050]))
    attach_held_objects([r], [far])
    assert r.holding == []


def test_elbow_stands_in_for_hidden_wrist(desk_person):
    r = result(desk_person(left="elbow_up"))
    phone = phone_near(r.person, KP.LEFT_ELBOW)
    attach_held_objects([r], [phone])
    assert r.holding == ["cell phone"]


def test_holding_in_payload(person_factory):
    r = result(person_factory())
    r.holding = ["cup"]
    assert r.to_dict()["holding"] == ["cup"]


def test_mirrored_object_box():
    obj = DetectedObject("cup", 0.9, np.array([10, 20, 30, 40]))
    assert obj.mirrored(100).bbox.tolist() == [70, 20, 90, 40]


def test_periodic_objects_reuses_results_between_runs():
    det = FakeObjectDetector([DetectedObject("cup", 0.9, np.array([0, 0, 1, 1]))])
    periodic = PeriodicObjects(det, every_n=3)
    outs = [periodic.update(None) for _ in range(6)]
    assert det.calls == 2
    assert all(o[0].label == "cup" for o in outs)


def test_periodic_objects_waits_until_ready():
    det = FakeObjectDetector([DetectedObject("cup", 0.9, np.array([0, 0, 1, 1]))])
    ready = False
    periodic = PeriodicObjects(det, ready=lambda: ready)
    assert periodic.update(None) == []
    ready = True
    assert periodic.update(None)[0].label == "cup"
    assert PeriodicObjects(None).update(None) == []


def test_large_objects_are_never_held(person_factory):
    r = result(person_factory(arms="down"))
    x1, y1, x2, y2 = r.person.bbox
    bus = DetectedObject("bus", 0.9, np.array([x1 - 300, y1 - 100, x2 + 300, y2]))
    attach_held_objects([r], [bus])
    assert r.holding == []
