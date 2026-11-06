import numpy as np

from posedemo.vision.tracking import IouTracker, iou
from posedemo.vision.types import Person


def person(x1, y1, x2, y2):
    return Person(np.zeros((17, 3)), np.array([x1, y1, x2, y2], dtype=float))


def test_iou():
    assert iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0
    assert iou(np.array([0, 0, 10, 10]), np.array([5, 0, 15, 10])) == 50 / 150


def test_ids_stable_across_frames():
    t = IouTracker()
    a, b = t.update([person(0, 0, 100, 200), person(300, 0, 400, 200)])
    assert (a.track_id, b.track_id) == (1, 2)
    # Same people, slightly moved and in reverse order.
    b2, a2 = t.update([person(305, 0, 405, 200), person(4, 0, 104, 200)])
    assert (a2.track_id, b2.track_id) == (1, 2)


def test_new_person_gets_new_id_and_lost_tracks_expire():
    t = IouTracker(max_age=2)
    t.update([person(0, 0, 100, 200)])
    for _ in range(3):
        t.update([])
    (p,) = t.update([person(0, 0, 100, 200)])
    assert p.track_id == 2


def test_brief_occlusion_keeps_id():
    t = IouTracker(max_age=5)
    t.update([person(0, 0, 100, 200)])
    t.update([])
    (p,) = t.update([person(2, 0, 102, 200)])
    assert p.track_id == 1
