from posedemo.poses import LabelSmoother
from posedemo.vision.types import PoseResult


def res(person, label, score=0.9):
    return PoseResult(person, label, score, {label: score, "standing": 0.7})


def test_majority_vote_suppresses_flicker(person_factory):
    p = person_factory(track_id=1)
    s = LabelSmoother(window=5)
    for _ in range(4):
        s.update([res(p, "standing")])
    out = s.update([res(p, "squat")])[0]
    assert out.label == "standing"
    assert out.score == 0.7


def test_untracked_people_pass_through(person_factory):
    p = person_factory(track_id=None)
    s = LabelSmoother(window=5)
    s.update([res(p, "standing")])
    assert s.update([res(p, "squat")])[0].label == "squat"


def test_stale_tracks_pruned(person_factory):
    s = LabelSmoother(window=3, max_missing=2)
    s.update([res(person_factory(track_id=7), "standing")])
    for _ in range(3):
        s.update([])
    assert 7 not in s._history
