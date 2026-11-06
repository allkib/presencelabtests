import numpy as np
import pytest

from posedemo.vision.geometry import angle, bell, distance, midpoint, ramp


@pytest.mark.parametrize(
    ("a", "b", "c", "expected"),
    [((1, 0), (0, 0), (0, 1), 90), ((1, 0), (0, 0), (-1, 0), 180), ((1, 0), (0, 0), (2, 0), 0)],
)
def test_angle(a, b, c, expected):
    assert angle(np.array(a), np.array(b), np.array(c)) == pytest.approx(expected)


def test_angle_degenerate():
    assert angle(np.zeros(2), np.zeros(2), np.ones(2)) == 0.0


def test_distance_midpoint():
    assert distance((0, 0), (3, 4)) == 5
    assert midpoint((0, 0), (2, 4)).tolist() == [1, 2]


def test_ramp_both_directions():
    assert ramp(5, 0, 10) == 0.5
    assert ramp(-1, 0, 10) == 0.0
    assert ramp(20, 0, 10) == 1.0
    assert ramp(130, 150, 110) == 0.5


def test_bell():
    assert bell(90, 90, 30) == 1.0
    assert bell(105, 90, 30) == 0.5
    assert bell(200, 90, 30) == 0.0
