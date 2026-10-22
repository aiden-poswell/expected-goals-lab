"""Shot geometry for attack-relative Wyscout coordinates. Pure functions, no data access.

Wyscout positions are integer percentages of the pitch from the acting team's point of view:
x = 100 is the goal being attacked and y = 50 is the centre line (see docs/GEOMETRY_VALIDATION.md).
No pitch dimensions are published with the data, so the metric conversion rests on an assumed
pitch (default 105 m x 68 m) and the laws-of-the-game goal width (7.32 m). Distances and angles
computed here are therefore approximations of the physical values, not measurements.

The attacked goal centre is (length, width / 2). The posts are at (length, width / 2 -/+ goal / 2).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pitch:
    length_m: float = 105.0
    width_m: float = 68.0
    goal_width_m: float = 7.32


REFERENCE_PITCH = Pitch()
ALTERNATIVE_PITCHES = {
    "100x64": Pitch(length_m=100.0, width_m=64.0),
    "110x70": Pitch(length_m=110.0, width_m=70.0),
}


@dataclass(frozen=True)
class ShotGeometry:
    x_m: float
    y_m: float
    distance_m: float
    angle_rad: float

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad)


def to_metres(x_pct: float, y_pct: float, pitch: Pitch = REFERENCE_PITCH) -> tuple[float, float]:
    """Attack-relative percentages to metres, with no flipping: x_m = L * x / 100."""
    _require_finite(x_pct, y_pct)
    if not (0 <= x_pct <= 100 and 0 <= y_pct <= 100):
        raise ValueError(f"position ({x_pct}, {y_pct}) is outside the 0-100 percentage range")
    return pitch.length_m * x_pct / 100, pitch.width_m * y_pct / 100


def distance_to_goal(x_m: float, y_m: float, pitch: Pitch = REFERENCE_PITCH) -> float:
    """Straight-line distance from the shot position to the centre of the goal mouth."""
    _require_on_pitch(x_m, y_m, pitch)
    return math.hypot(pitch.length_m - x_m, pitch.width_m / 2 - y_m)


def goal_angle(x_m: float, y_m: float, pitch: Pitch = REFERENCE_PITCH) -> float:
    """Angle in radians, in [0, pi], subtended at the shot position by the two goalposts.

    Computed as atan2(|cross|, dot) of the vectors to the posts, which stays accurate for shots
    on the centre line (where it equals 2*atan(goal / (2*depth))), close to the goal line and far
    out. On the goal line between the posts the angle is pi; on the goal line outside the posts
    it is 0. A position exactly on a post has no defined angle and raises ValueError.
    """
    _require_on_pitch(x_m, y_m, pitch)
    depth = pitch.length_m - x_m
    to_post_a = (depth, pitch.width_m / 2 - pitch.goal_width_m / 2 - y_m)
    to_post_b = (depth, pitch.width_m / 2 + pitch.goal_width_m / 2 - y_m)
    if to_post_a == (0.0, 0.0) or to_post_b == (0.0, 0.0):
        raise ValueError(f"position ({x_m}, {y_m}) coincides with a goalpost")
    cross = to_post_a[0] * to_post_b[1] - to_post_a[1] * to_post_b[0]
    dot = to_post_a[0] * to_post_b[0] + to_post_a[1] * to_post_b[1]
    return math.atan2(abs(cross), dot)


def shot_geometry(x_pct: float, y_pct: float, pitch: Pitch = REFERENCE_PITCH) -> ShotGeometry:
    x_m, y_m = to_metres(x_pct, y_pct, pitch)
    return ShotGeometry(
        x_m=x_m,
        y_m=y_m,
        distance_m=distance_to_goal(x_m, y_m, pitch),
        angle_rad=goal_angle(x_m, y_m, pitch),
    )


def _require_finite(*values: float) -> None:
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"non-finite coordinate in {values}")


def _require_on_pitch(x_m: float, y_m: float, pitch: Pitch) -> None:
    _require_finite(x_m, y_m)
    if not (0 <= x_m <= pitch.length_m and 0 <= y_m <= pitch.width_m):
        raise ValueError(f"position ({x_m}, {y_m}) m is outside the {pitch} pitch")
