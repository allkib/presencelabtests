import pytest

from posedemo.poses import UNKNOWN, PoseClassifier, all_rules, register
from posedemo.poses.base import PoseRule
from posedemo.vision.keypoints import KP


@pytest.fixture
def clf():
    return PoseClassifier(threshold=0.6, kp_conf=0.5)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"arms": "down"}, "standing"),
        ({"arms": "out"}, "t_pose"),
        ({"arms": "up"}, "hands_up"),
        ({"arms": "left_up"}, "left_hand_up"),
        ({"arms": "right_up"}, "right_hand_up"),
        ({"arms": "crossed"}, "arms_crossed"),
        ({"arms": "down", "squat": True}, "squat"),
    ],
)
def test_builtin_poses(clf, person_factory, kwargs, expected):
    result = clf.classify_one(person_factory(**kwargs))
    assert result.label == expected, result.scores
    assert result.score >= 0.6


def test_invisible_keypoints_give_unknown(clf, person_factory):
    p = person_factory()
    p.keypoints[:, 2] = 0.1
    assert clf.classify_one(p).label == UNKNOWN


def test_upper_body_only_still_works(clf, person_factory):
    p = person_factory(arms="up")
    for k in (KP.LEFT_KNEE, KP.RIGHT_KNEE, KP.LEFT_ANKLE, KP.RIGHT_ANKLE):
        p.keypoints[k, 2] = 0.0
    assert clf.classify_one(p).label == "hands_up"


def test_scale_invariance(clf, person_factory):
    p = person_factory(arms="out")
    p.keypoints[:, :2] *= 0.3
    assert clf.classify_one(p).label == "t_pose"


def test_registry_rejects_duplicates():
    names = [r.name for r in all_rules()]
    assert len(names) == len(set(names))
    with pytest.raises(ValueError):

        @register
        class Dup(PoseRule):
            name = names[0]

            def score(self, person):
                return 1.0


def test_custom_rule_via_injection(person_factory):
    class Always(PoseRule):
        name = "always"

        def score(self, person):
            return 0.9

    clf = PoseClassifier(rules=[Always()])
    assert clf.classify_one(person_factory()).label == "always"


HIDDEN_LOWER_BODY = (
    KP.LEFT_HIP, KP.RIGHT_HIP, KP.LEFT_KNEE, KP.RIGHT_KNEE, KP.LEFT_ANKLE, KP.RIGHT_ANKLE,
)  # fmt: skip


@pytest.mark.parametrize(
    ("arms", "expected"),
    [
        ("out", "t_pose"),
        ("up", "hands_up"),
        ("left_up", "left_hand_up"),
        ("right_up", "right_hand_up"),
        ("crossed", "arms_crossed"),
        ("down", UNKNOWN),
    ],
)
def test_webcam_framing_without_hips(clf, person_factory, arms, expected):
    """Sitting close to a webcam: only head, shoulders and arms are in frame."""
    p = person_factory(arms=arms)
    for k in HIDDEN_LOWER_BODY:
        p.keypoints[k, 2] = 0.05
    result = clf.classify_one(p)
    assert result.label == expected, result.scores
