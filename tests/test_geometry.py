"""Geometry tests against analytic values, not against the implementation's own formula.

The independent reference is the law of cosines on the triangle (shot, post A, post B):
cos(angle) = (da^2 + db^2 - g^2) / (2 * da * db), with da, db the distances to the posts.
"""

import itertools
import math

import pytest

from expected_goals_lab.geometry import (
    ALTERNATIVE_PITCHES,
    REFERENCE_PITCH,
    Pitch,
    distance_to_goal,
    goal_angle,
    shot_geometry,
    to_metres,
)

L, W, G = 105.0, 68.0, 7.32
POST_Y = (W / 2 - G / 2, W / 2 + G / 2)


def law_of_cosines_angle(x_m: float, y_m: float) -> float:
    da = math.hypot(L - x_m, POST_Y[0] - y_m)
    db = math.hypot(L - x_m, POST_Y[1] - y_m)
    return math.acos((da**2 + db**2 - G**2) / (2 * da * db))


def test_conversion_is_linear_and_never_flips():
    assert to_metres(0, 0) == (0.0, 0.0)
    assert to_metres(50, 50) == (52.5, 34.0)
    assert to_metres(100, 100) == (105.0, 68.0)
    assert to_metres(89.5238095, 50)[0] == pytest.approx(94.0, abs=1e-4)
    # x = 100 is the attacked goal line for every team and period: larger x is always closer.
    assert distance_to_goal(*to_metres(90, 50)) < distance_to_goal(*to_metres(10, 50))


@pytest.mark.parametrize(
    ("depth", "expected_deg"), [(11.0, 36.8074), (5.5, 67.2839), (22.0, 18.8909)]
)
def test_centre_line_angle_matches_closed_form(depth, expected_deg):
    # expected degrees: 2 * atan(3.66 / depth), computed separately from the implementation
    angle = goal_angle(L - depth, W / 2)
    assert angle == pytest.approx(2 * math.atan(G / (2 * depth)), abs=1e-12)
    assert math.degrees(angle) == pytest.approx(expected_deg, abs=1e-3)


def test_penalty_spot_opening_angle_is_about_36_8_degrees():
    geometry = shot_geometry(100 * 94.0 / L, 50)
    assert geometry.distance_m == pytest.approx(11.0, abs=1e-9)
    assert geometry.angle_deg == pytest.approx(36.8, abs=0.05)


@pytest.mark.parametrize(
    ("x_m", "y_m"),
    [(94, 20), (94, 48), (80, 5), (100, 1), (104, 30), (60, 0.0), (70, 68.0), (99, 33), (50, 34)],
)
def test_angle_agrees_with_law_of_cosines_off_centre_and_near_touchline(x_m, y_m):
    assert goal_angle(x_m, y_m) == pytest.approx(law_of_cosines_angle(x_m, y_m), abs=1e-9)


def test_distance_uses_the_goal_centre():
    assert distance_to_goal(L - 12, W / 2) == pytest.approx(12.0)
    assert distance_to_goal(L - 4, W / 2 - 3) == pytest.approx(5.0)  # 3-4-5 triangle
    assert distance_to_goal(0, 0) == pytest.approx(math.hypot(L, W / 2))


@pytest.mark.parametrize("offset", [0.5, 3.0, 10.0, 33.9])
def test_mirrored_y_positions_give_identical_geometry(offset):
    above = shot_geometry(88, 50 + offset / W * 100)
    below = shot_geometry(88, 50 - offset / W * 100)
    assert above.distance_m == pytest.approx(below.distance_m, abs=1e-12)
    assert above.angle_rad == pytest.approx(below.angle_rad, abs=1e-12)


def test_angle_increases_towards_goal_on_the_centre_line():
    angles = [goal_angle(x, W / 2) for x in (40, 60, 80, 94, 103)]
    assert angles == sorted(angles)


def test_angle_decreases_for_wider_positions_at_fixed_depth():
    angles = [goal_angle(94, W / 2 + lateral) for lateral in (0, 2, 5, 10, 20, 34)]
    assert angles == sorted(angles, reverse=True)
    assert angles[0] > angles[-1] > 0


def test_goal_line_edge_cases():
    assert goal_angle(L, W / 2) == pytest.approx(math.pi)  # between the posts
    assert goal_angle(L, W / 2 + 3.0) == pytest.approx(math.pi)
    assert goal_angle(L, W / 2 + 5.0) == pytest.approx(0.0)  # on the line, outside the posts
    assert goal_angle(L, 1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="goalpost"):
        goal_angle(L, W / 2 - G / 2)


def test_near_goal_line_values_are_accurate_not_rounded_away():
    close = goal_angle(L - 1e-3, W / 2)
    assert 3.1 < close < math.pi
    assert close == pytest.approx(2 * math.atan(G / 2 / 1e-3), abs=1e-9)


def test_angle_is_always_within_zero_and_pi_on_a_full_grid():
    for x, y in itertools.product(range(0, 101), range(0, 101)):
        angle = goal_angle(*to_metres(x, y))
        assert 0.0 <= angle <= math.pi
        assert math.isfinite(distance_to_goal(*to_metres(x, y)))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_coordinates_are_rejected(bad):
    with pytest.raises(ValueError, match="non-finite"):
        to_metres(bad, 50)
    with pytest.raises(ValueError, match="non-finite"):
        to_metres(50, bad)
    with pytest.raises(ValueError, match="non-finite"):
        goal_angle(bad, 34)
    with pytest.raises(ValueError, match="non-finite"):
        distance_to_goal(50, bad)


@pytest.mark.parametrize("position", [(-1, 50), (101, 50), (50, -0.1), (50, 100.1)])
def test_positions_outside_the_percentage_range_are_rejected(position):
    with pytest.raises(ValueError, match="outside"):
        to_metres(*position)


def test_positions_outside_the_pitch_in_metres_are_rejected():
    with pytest.raises(ValueError, match="outside"):
        goal_angle(L + 1, 34)
    with pytest.raises(ValueError, match="outside"):
        distance_to_goal(50, -1)


def test_alternative_pitches_change_scale_but_keep_the_goal_width():
    reference = shot_geometry(90, 40)
    for pitch in ALTERNATIVE_PITCHES.values():
        assert pitch.goal_width_m == REFERENCE_PITCH.goal_width_m == 7.32
        other = shot_geometry(90, 40, pitch)
        assert (other.x_m, other.y_m) != (reference.x_m, reference.y_m)
    custom = Pitch(length_m=100.0, width_m=64.0)
    # Same percentages, shorter pitch: the shot is closer to goal along the pitch.
    assert shot_geometry(90, 50, custom).distance_m == pytest.approx(10.0)
    assert shot_geometry(90, 50).distance_m == pytest.approx(10.5)
