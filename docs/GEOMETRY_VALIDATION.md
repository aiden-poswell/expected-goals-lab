# Geometry validation (S02)

Code: `src/expected_goals_lab/geometry.py` (pure functions, no data access). Tests: `tests/test_geometry.py`. Figures from the data are from `reports/shot_dataset_audit.md`.

## 1. Conventions and assumptions

- **Source frame.** Wyscout start positions are integer percentages from the acting team's point of view. The documentation (<https://support.wyscout.com/matches-wyid-events>) says "percentage from the left corner of the attacking team"; the paper adds that x measures nearness to the opponent's goal. In the data, in-play penalties average x = 89.3, y = 50.0 and goal-tag share rises with x (S01B, `docs/WYSCOUT_DATA_AUDIT.md`). So: **x = 100 is the goal being attacked and y = 50 is the centre line, for both teams in both halves. Nothing is flipped by team, half or match.** The documentation does not settle which touchline y = 0 is. Distance and angle are symmetric about y = 50, so this does not affect them; it would affect any left/right feature.
- **Pitch.** The data carry no pitch dimensions. The reference is the usual 105 m x 68 m with a 7.32 m goal (the width fixed by the Laws of the Game); `x_m = 105 * x_pct / 100`, `y_m = 68 * y_pct / 100`, goal centre `(105, 34)`, posts at `y = 34 -/+ 3.66`. Real pitches vary, so the distances are approximations, not measurements. The reference stays 105 x 68 for S03.
- **Resolution.** Positions are integers, so one unit is about 1.05 m along the pitch and 0.68 m across it. Close to goal this quantisation matters most for the angle.

## 2. Definitions

- `distance_to_goal` = straight-line distance from the shot to the goal centre.
- `goal_angle` = the angle subtended at the shot by the two posts, in radians in `[0, pi]`, computed as `atan2(|cross|, dot)` of the vectors from the shot to the two posts. This is not `atan` of the lateral offset. On the centre line at depth `d` it equals `2 * atan(g / (2 d))` with `g = 7.32`.
- Edge cases: on the goal line between the posts the angle is pi; on the goal line outside the posts it is 0; exactly on a post it has no defined value and raises `ValueError`. Non-finite values, percentages outside 0-100, and positions outside the pitch raise `ValueError`. In the data, 7 shots are at `x_pct = 100` (angle 180 or 0 degrees); they carry the flag `ON_GOAL_LINE` and are not excluded. 242 eligible shots have an angle of 90 degrees or more, which is geometrically correct close to goal.

## 3. Independent validation

The tests compare the implementation with values it does not share code with:

| Check | Reference | Result |
|---|---|---|
| Centre line, depth 11 m (penalty spot) | closed form `2 atan(3.66/11)` = 36.8074 degrees | match to 1e-3 degrees |
| Centre line, depth 5.5 m and 22 m | closed form: 67.2839 and 18.8909 degrees | match |
| Off-centre, near-touchline, near-goal-line and mid-pitch points (nine) | law of cosines on the triangle shot-post-post, `cos(angle) = (da^2 + db^2 - g^2) / (2 da db)` | match to 1e-9 rad |
| Distance | 12 m straight in front; 3-4-5 triangle (depth 4, offset 3) gives 5 m; corner flag to goal centre | exact |
| Mirror symmetry | y and 68 - y positions | identical distance and angle |
| Monotonicity | angle increases towards goal on the centre line; decreases with lateral offset at fixed depth | holds |
| Goal-line cases | between posts pi, outside 0, on post raises | as specified |
| Whole 101 x 101 percentage grid | angle in `[0, pi]`, distance finite | holds |
| Direction | larger `x_pct` is always closer to goal; no flip | holds |
| Rejection | NaN, +/-inf, out-of-range percentages and metres | `ValueError` |

The penalty spot opening angle is 36.8 degrees (36.8074 exactly), agreeing with the figure of about 36.8 degrees.

## 4. Sensitivity to the assumed pitch size

Recomputed from the same percentages for the 40,425 eligible rows with the goal fixed at 7.32 m (`reports/shot_dataset_audit.md`, section 5):

| Pitch | Quantity | Mean abs diff | Max abs diff | Spearman rho vs 105 x 68 | Max percentile shift | Share of shots moving > 0.05 |
|---|---|---|---|---|---|---|
| 100 x 64 | distance (m) | 0.930 | 3.751 | 0.99996 | 0.0165 | 0.00% |
| 100 x 64 | angle (degrees) | 1.287 | 11.728 | 0.99979 | 0.0627 | 0.03% |
| 110 x 70 | distance (m) | 0.801 | 3.750 | 0.99994 | 0.0177 | 0.00% |
| 110 x 70 | angle (degrees) | 0.887 | 5.907 | 0.99984 | 0.0205 | 0.00% |

Eligible-shot distance: median 17.1 m (quartiles 12.2 and 24.9). Angle: median 18.8 degrees (quartiles 14.4 and 29.0).

Reading: absolute distances move by about a metre on average and by up to about 3.8 m, and angles by about one degree on average and by up to 11.7 degrees for the most extreme close shot, so metre values should not be quoted as exact. The ranking of shots is almost unchanged (Spearman above 0.9997; at most 0.03% of shots move more than five percentile points). A model that uses these quantities monotonically should therefore be insensitive to the choice within this range, though a model that is sensitive to absolute scale, or the calibration of a fitted coefficient, is not guaranteed to be; S03 should refit under at least one alternative pitch and say so.

## 5. Not established

- The true pitch of any match.
- Which touchline `y = 0` is.
- Whether a position recorded at `x_pct` of 20 or more is ever recorded in a reversed frame (the 18 quarantined suspect positions were found through implausible location, not through a general test).
