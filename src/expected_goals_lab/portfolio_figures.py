"""S08 portfolio figures: held-out evidence, shot quality across the pitch, one real match.

    uv run python -m expected_goals_lab.portfolio_figures

Figure 1 reads the frozen S07 summary and computes nothing. Figures 2 and 3 use the model fitted
on the original training split only (the unchanged S04 pipeline); Figure 3 shows one match of the
original validation window. No held-out feature, outcome or prediction is inspected or used
(the shot table is still traversed for identity columns, as in S03-S05). Rules:
docs/S08_VISUAL_SPEC.md.
"""

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from expected_goals_lab import portfolio_data as pdata
from expected_goals_lab.candidate_data import CandidatePartition
from expected_goals_lab.common import DataAccessError
from expected_goals_lab.dataset import DEFAULT_CSV, Partition, iter_csv_rows
from expected_goals_lab.evaluate_baselines import LEAGUE_NAMES
from expected_goals_lab.evaluate_nonlinear import (
    DEFAULT_PROTOCOL,
    DEFAULT_S03_REPORT,
    evaluate_nonlinear,
)
from expected_goals_lab.figures import (
    ATTRIBUTION,
    BLUE,
    GREY,
    INK,
    LIGHT,
    ORANGE,
    PNG_DPI,
    _caption,
    _headline,
    _save,
    _signed,
    _style,
    plt,
)
from expected_goals_lab.final_evaluation import recipe_sha256
from expected_goals_lab.final_lock import REAL_LOCKED
from expected_goals_lab.geometry import REFERENCE_PITCH, distance_to_goal, goal_angle
from expected_goals_lab.nonlinear import NONLINEAR_LOCATION_BODY_PART
from expected_goals_lab.splits import split_of_events
from expected_goals_lab.wyscout import WyscoutSource

DEFAULT_OUT = Path("figures/portfolio")
VALIDATION_SUMMARY = Path("data/processed/nonlinear_validation_summary.json")
RECORDED_MATCH_ID = 2516947  # docs/S08_VISUAL_SPEC.md, section 5
GRID_STEP_M = 0.5
GRID_X_FROM_M = 60.0
STEMS = ("fig1_heldout_evidence", "fig2_shot_quality_map", "fig3_match_shot_map")
BUILD_COMMAND = "uv run python -m expected_goals_lab.portfolio_figures"
MODEL_LABELS = {
    pdata.PRIMARY_KEY: "Location + body-part tag\n(selected model)",
    pdata.BENCHMARK_KEY: "Distance + angle\n(benchmark)",
    pdata.CONSTANT_KEY: "Same chance for every shot\n(reference)",
}
COLOUR_CAP = 0.50
CMAP = "YlGnBu"
PAPER_NOTE = (
    "Paper: Pappalardo et al. (2019), Scientific Data 6:236, doi:10.1038/s41597-019-0247-7."
)


# --- pitch  -----------------------------------------------------------------------------------


def pitch_markings(length: float = 105.0, width: float = 68.0) -> dict:
    """Laws-of-the-game dimensions in metres, used by ``draw_pitch`` and checked by tests."""
    return {
        "penalty_area": (16.5, 40.32),
        "six_yard_box": (5.5, 18.32),
        "penalty_spot": 11.0,
        "circle_radius": 9.15,
        "goal_width": 7.32,
        "centre": (length / 2, width / 2),
    }


def draw_pitch(ax, x_from: float = 0.0, line: str = "#7b8088") -> None:
    """Draw a pitch (or its right-hand part from ``x_from``) with the attacked goal at x = 105."""
    from matplotlib.patches import Arc, Circle, Rectangle

    length, width = REFERENCE_PITCH.length_m, REFERENCE_PITCH.width_m
    m = pitch_markings(length, width)
    ax.add_patch(Rectangle((0, 0), length, width, fc="#f3f5f0", ec=line, lw=1.4, zorder=0))
    ax.plot([length / 2] * 2, [0, width], color=line, lw=1.1, zorder=1)
    ax.add_patch(Circle(m["centre"], m["circle_radius"], fill=False, ec=line, lw=1.1, zorder=1))
    ax.scatter(*m["centre"], s=10, color=line, zorder=1)
    for sign in (1, -1):  # right-hand end, then left-hand end
        goal_x = length if sign == 1 else 0.0

        def edge(depth: float, goal_x=goal_x, sign=sign) -> float:
            return goal_x - sign * depth

        for depth, span, lw in (
            (m["penalty_area"][0], m["penalty_area"][1], 1.1),
            (m["six_yard_box"][0], m["six_yard_box"][1], 1.1),
        ):
            ax.add_patch(
                Rectangle(
                    (min(goal_x, edge(depth)), (width - span) / 2),
                    depth,
                    span,
                    fill=False,
                    ec=line,
                    lw=lw,
                    zorder=1,
                )
            )
        spot_x = edge(m["penalty_spot"])
        ax.scatter(spot_x, width / 2, s=10, color=line, zorder=1)
        half_angle = np.degrees(np.arccos((m["penalty_area"][0] - m["penalty_spot"]) / 9.15))
        centre_angle = 180.0 if sign == 1 else 0.0
        ax.add_patch(
            Arc(
                (spot_x, width / 2),
                2 * 9.15,
                2 * 9.15,
                theta1=centre_angle - half_angle,
                theta2=centre_angle + half_angle,
                ec=line,
                lw=1.1,
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (goal_x if sign == 1 else goal_x - 2.0, (width - m["goal_width"]) / 2),
                2.0,
                m["goal_width"],
                fc="white",
                ec=line,
                lw=1.4,
                zorder=1,
            )
        )
    ax.set_aspect("equal")
    ax.set_xlim(-3.0 if x_from <= 0 else x_from - 1.0, length + 3.0)
    ax.set_ylim(width + 1.5, -1.5)  # y grows downwards, as in the provider's convention
    ax.axis("off")


# --- Figure 1  --------------------------------------------------------------------------------


def _note(fig, ax, text: str, width: int = 100) -> None:
    """A wrapped note under an axis, aligned with the figure's left text margin."""
    y = ax.get_position().y0 - 0.052
    fig.text(0.012, y, textwrap.fill(text, width), fontsize=9, color=INK, ha="left", va="top")


def _pretty_date(iso: str) -> str:
    d = date.fromisoformat(iso)
    return f"{d.day} {d:%b %Y}"


def plot_heldout_evidence(stem: Path, h: dict) -> list[Path]:
    """Log loss, Brier score and summed xG for the three models, from the frozen S07 summary."""
    _style()
    keys = [pdata.PRIMARY_KEY, pdata.BENCHMARK_KEY, pdata.CONSTANT_KEY]
    colours = {pdata.PRIMARY_KEY: BLUE, pdata.BENCHMARK_KEY: GREY, pdata.CONSTANT_KEY: "#bfc4cc"}
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 11.0), gridspec_kw={"height_ratios": [1, 1, 1]})
    fig.subplots_adjust(left=0.30, right=0.93, top=0.835, bottom=0.215, hspace=0.95)
    _headline(
        fig,
        "On unseen later shots, location + body part scored better than distance + angle",
        f"{h['shots']:,} regular shots, {h['matches']} matches, five leagues, "
        f"{_pretty_date(h['first_date'])} to {_pretty_date(h['last_date'])}. "
        "One held-out evaluation; smaller is better for both scores.",
    )
    panels = [
        ("log_loss", "Log loss", "lower is better", 4),
        ("brier", "Brier score", "lower is better", 5),
    ]
    rows = np.arange(len(keys))[::-1]
    for ax, (metric, title, direction, digits) in zip(axes[:2], panels, strict=False):
        values = [h["scores"][k][metric] for k in keys]
        low, high = min(values), max(values)
        pad = (high - low) * 0.22
        ax.set_xlim(low - pad * 0.6, high + pad)
        ax.set_ylim(-0.6, len(keys) - 0.4)
        for row, key, value in zip(rows, keys, values, strict=True):
            ax.plot([ax.get_xlim()[0], value], [row, row], color=LIGHT, lw=1.2, zorder=1)
            ax.scatter([value], [row], s=130, color=colours[key], zorder=3)
            ax.annotate(
                f"{value:.{digits}f}",
                (value, row),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=10.5,
                fontweight="bold",
                color=colours[key] if key != pdata.CONSTANT_KEY else "#6c727c",
            )
        ax.set_yticks(rows, [MODEL_LABELS[k] for k in keys])
        ax.tick_params(axis="y", length=0, labelsize=9.5)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.spines["left"].set_visible(False)
        ax.set_title(f"{title}  ({direction})", loc="left", fontsize=11)
        ax.set_xlabel(
            f"axis starts at {ax.get_xlim()[0]:.{digits - 1}f}, not at zero", fontsize=8.5
        )
        iv = h["intervals"][metric]
        rel = h["relative_log_loss"] if metric == "log_loss" else h["relative_brier"]
        _note(
            fig,
            ax,
            f"Selected minus benchmark: {_signed(iv['estimate'], digits)} "
            f"({_signed(100 * rel, 1)}%). 95% match-bootstrap interval: "
            f"{_signed(iv['low'], digits)} to {_signed(iv['high'], digits)}.",
        )
    ax = axes[2]
    goals = h["goals"]
    bars = [(pdata.PRIMARY_KEY, "Selected model"), (pdata.BENCHMARK_KEY, "Distance + angle")]
    names = ["Goals scored"] + [label for _, label in bars]
    values = [goals] + [h["scores"][k]["expected_goals"] for k, _ in bars]
    colours3 = [ORANGE, BLUE, GREY]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, values, color=colours3, height=0.55, zorder=2)
    for y, value, name in zip(ypos, values, names, strict=True):
        text = f"{value:,.0f}" if name == "Goals scored" else f"{value:,.1f} expected"
        ax.annotate(
            text,
            (value, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_yticks(ypos, names)
    ax.tick_params(axis="y", length=0, labelsize=9.5)
    ax.set_xlim(0, goals * 1.18)
    ax.spines["left"].set_visible(False)
    ax.set_title("Context: goals scored and goals expected over all shots", loc="left", fontsize=11)
    ax.set_xlabel("bars start at zero", fontsize=8.5)
    _note(
        fig,
        ax,
        "Summed xG is each model's total of shot probabilities. Both models expect slightly "
        "fewer goals than were scored; this describes the total, not any player or team.",
    )
    _caption(
        fig,
        "Scores are those of the single held-out evaluation, read from its frozen summary and "
        "not recomputed. Intervals are exploratory: a paired bootstrap over whole matches that "
        "does not cover the choice among candidate models, dependence between matches of a team, "
        "or other seasons. The two models differ in their location terms as well as in the "
        "body-part tag, so the gap is not an ablation of body part alone. "
        f"{PAPER_NOTE} {ATTRIBUTION}",
        width=150,
    )
    return _save(fig, stem)


# --- Figure 2  --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbabilityGrid:
    x_m: np.ndarray  # cell centres
    y_m: np.ndarray
    probability: dict[str, np.ndarray]  # class -> (len(y), len(x)); NaN where unsupported
    support: dict[str, np.ndarray]  # class -> training shots nearby


def grid_axes(step: float = GRID_STEP_M, x_from: float = GRID_X_FROM_M) -> tuple:
    length, width = REFERENCE_PITCH.length_m, REFERENCE_PITCH.width_m
    return (
        np.arange(x_from + step / 2, length, step),
        np.arange(step / 2, width, step),
    )


def _grid_partition(x_m: np.ndarray, y_m: np.ndarray, body_part: str) -> CandidatePartition:
    xs, ys = np.meshgrid(x_m, y_m)
    xs, ys = xs.ravel(), ys.ravel()
    n = len(xs)
    distance = np.array([distance_to_goal(x, y) for x, y in zip(xs, ys, strict=True)])
    angle = np.array([goal_angle(x, y) for x, y in zip(xs, ys, strict=True)])
    base = Partition(
        split="train",
        event_ids=np.arange(n),
        match_ids=np.zeros(n, dtype=int),
        competition_ids=np.zeros(n, dtype=int),
        dates=np.array(["2000-01-01"] * n),
        features={"distance_m": distance, "angle_rad": angle},
        x_pct=np.zeros(n, dtype=int),
        y_pct=np.zeros(n, dtype=int),
        y=np.zeros(n, dtype=np.int8),
    )
    return CandidatePartition(base=base, body_part=np.array([body_part] * n))


def probability_grid(candidate, train: CandidatePartition) -> ProbabilityGrid:
    """Estimates of the training-only model on a regular grid, blank where training is thin."""
    x_m, y_m = grid_axes()
    train_x, train_y = pdata.pct_to_metres(train.base.x_pct, train.base.y_pct)
    probability, support = {}, {}
    for cls in pdata.BODY_PART_LABELS:
        of_class = train.body_part == cls
        counts = pdata.support_counts(train_x[of_class], train_y[of_class], x_m, y_m)
        p = candidate.predict_proba(_grid_partition(x_m, y_m, cls)).reshape(len(y_m), len(x_m))
        probability[cls] = np.where(counts >= pdata.SUPPORT_MIN_SHOTS, p, np.nan)
        support[cls] = counts
    return ProbabilityGrid(x_m, y_m, probability, support)


def plot_shot_quality_map(stem: Path, grid: ProbabilityGrid, n_train: int) -> list[Path]:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 8.4))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.865, bottom=0.215, wspace=0.03)
    _headline(
        fig,
        "Estimated shot chance rises sharply near goal and centrally; the body-part tag moves it",
        "Estimated chance of a goal for a regular shot from each spot, attacking the goal on the "
        "right. Same colour scale in all panels.",
    )
    vmax = COLOUR_CAP  # point-blank positions exceed this; they are drawn in the darkest colour
    mesh = None
    for ax, cls in zip(axes, pdata.BODY_PART_LABELS, strict=True):
        draw_pitch(ax, x_from=GRID_X_FROM_M)
        p = np.ma.masked_invalid(grid.probability[cls])
        mesh = ax.pcolormesh(
            grid.x_m,
            grid.y_m,
            p,
            cmap=CMAP,
            vmin=0,
            vmax=vmax,
            shading="nearest",
            zorder=0.5,
            rasterized=True,
        )
        if p.count():
            cs = ax.contour(
                grid.x_m,
                grid.y_m,
                p,
                levels=[0.05, 0.10, 0.20, 0.30],
                colors="#1d2733",
                linewidths=0.7,
                zorder=3,
            )
            ax.clabel(
                cs, fmt=lambda v: f"{100 * v:.0f}%", fontsize=8.5, inline=True, inline_spacing=8
            )
        ax.set_title(pdata.BODY_PART_LABELS[cls], loc="left", fontsize=11.5, pad=6)
    cax = fig.add_axes([0.30, 0.165, 0.40, 0.018])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("Estimated chance of a goal (50% or more in the darkest colour)", fontsize=9)
    ticks = np.arange(0, vmax + 1e-9, 0.10)
    cbar.ax.set_xticks(ticks)
    cbar.ax.set_xticklabels([f"{100 * v:.0f}%" for v in ticks])
    cbar.ax.tick_params(labelsize=8.5)
    _caption(
        fig,
        "Blank = fewer than "
        f"{pdata.SUPPORT_MIN_SHOTS} training shots of that kind within 3 m (in each direction), "
        "so no estimate is drawn. The model is the selected location + body-part recipe fitted on "
        f"the original training split only ({n_train:,} shots, matches before 2018-02-26): no "
        "validation or test shot was used. Positions assume a 105 m x 68 m pitch. Differences "
        "between panels describe how shots tagged this way differ in the data, not what changing "
        'how a shot is struck would do; "head / other body" is Wyscout tag 403, not a confirmed '
        f"header. {ATTRIBUTION}",
        width=165,
    )
    return _save(fig, stem)


# --- Figure 3  --------------------------------------------------------------------------------

XG_AREA_PER_UNIT = 800.0  # display scale only; reduced in S08A so markers do not cover the goal
ORIGIN_DOT_AREA = 9.0  # the exact recorded position of every shot
MATCH_FIG_SIZE = (11.0, 9.0)
MATCH_AXES_RECT = [0.03, 0.215, 0.94, 0.56]
MATCH_YLIM = (REFERENCE_PITCH.width_m + 1.5, -8.0)  # marker area (points squared) per unit of xG


@dataclass(frozen=True)
class MatchShots:
    info: pdata.MatchInfo
    competition: str
    side: np.ndarray
    team_id: np.ndarray
    x_pct: np.ndarray
    y_pct: np.ndarray
    goal: np.ndarray
    xg: np.ndarray
    totals: dict[int, dict]


def _marker_key(fig, rect: list[float]) -> None:
    """Legend drawn in figure space so it uses the same area scale as the data."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    width_pt = rect[2] * fig.get_figwidth() * 72
    units = 12.0
    pt_per_unit = width_pt / units
    ax.set_xlim(0, units)
    ax.set_ylim(0, 1)
    x = 0.2
    ax.text(x, 0.5, "Marker area = chance:", fontsize=9, va="center", color=INK)
    x += 3.1
    for value in (0.05, 0.20, 0.50):
        radius = np.sqrt(XG_AREA_PER_UNIT * value / np.pi) / pt_per_unit
        x += radius
        ax.scatter([x], [0.5], s=XG_AREA_PER_UNIT * value, facecolor="#9aa0a6", alpha=0.45,
                   edgecolor="#555555", linewidth=1.4, clip_on=False)  # fmt: skip
        ax.text(x + radius + 0.12, 0.5, f"{100 * value:.0f}%", fontsize=9, va="center", color=INK)
        x += radius + 0.9
    ax.scatter([x], [0.5], s=170, facecolor="#9aa0a6", edgecolor="#111111", linewidth=1.6)
    ax.text(x + 0.25, 0.5, "goal", fontsize=9, va="center", color=INK)
    x += 1.4
    ax.scatter(
        [x], [0.5], s=170, facecolor="#9aa0a6", alpha=0.45, edgecolor="#555555", linewidth=1.4
    )
    ax.text(x + 0.25, 0.5, "no goal", fontsize=9, va="center", color=INK)


def plot_match_shot_map(stem: Path, m: MatchShots) -> list[Path]:
    _style()
    fig = plt.figure(figsize=MATCH_FIG_SIZE)
    ax = fig.add_axes(MATCH_AXES_RECT)
    draw_pitch(ax)
    home_t, away_t = m.totals[m.info.home_id], m.totals[m.info.away_id]
    colour = {"home": BLUE, "away": ORANGE}
    for side in ("away", "home"):
        sel = m.side == side
        x, y = pdata.to_display(m.x_pct[sel], m.y_pct[sel], side)
        size = XG_AREA_PER_UNIT * m.xg[sel]
        scored = m.goal[sel] == 1
        ax.scatter(x[~scored], y[~scored], s=size[~scored], facecolor=colour[side],
                   edgecolor=colour[side], alpha=0.38, linewidth=1.6, zorder=3)  # fmt: skip
        ax.scatter(x[scored], y[scored], s=size[scored], facecolor=colour[side],
                   edgecolor="#111111", linewidth=1.8, alpha=0.95, zorder=4)  # fmt: skip
        ax.scatter(x, y, s=ORIGIN_DOT_AREA, color="#111111", zorder=6, linewidths=0)
    for centre, key, name in (
        (52.5 + 22, "home", m.info.home_name),
        (52.5 - 22, "away", m.info.away_name),
    ):
        sign = 1 if key == "home" else -1
        ax.annotate("", xy=(centre + sign * 11, -3.2), xytext=(centre - sign * 11, -3.2),
                    arrowprops={"arrowstyle": "-|>", "color": colour[key], "lw": 1.8})  # fmt: skip
        ax.text(centre, -5.0, f"{name} attack", color=colour[key], fontsize=10, fontweight="bold",
                ha="center", va="bottom")  # fmt: skip
    ax.set_ylim(*MATCH_YLIM)

    kickoff = m.info.date_utc[:10]
    _headline(
        fig,
        f"{m.info.home_name} v {m.info.away_name}: the shots as the model saw them",
        f"{m.competition}, {_pretty_date(kickoff)}. A validation-window match, scored by a model "
        "fitted on earlier training matches only.\nMarker area = the model's estimated chance of "
        "a goal (scale reduced for legibility); small black dot = recorded shot position;\n"
        "filled with a dark outline = scored.",
    )
    _marker_key(fig, [0.06, 0.79, 0.88, 0.035])
    for x_text, ha, name, tot, key in (
        (0.955, "right", m.info.home_name, home_t, "home"),
        (0.045, "left", m.info.away_name, away_t, "away"),
    ):
        side_word = "home, attacking right" if key == "home" else "away, attacking left"
        fig.text(x_text, 0.188, name, color=colour[key], fontsize=13, fontweight="bold", ha=ha)
        fig.text(x_text, 0.163, f"({side_word})", color="#555555", fontsize=9, ha=ha)
        plural = "shot" if tot["shots"] == 1 else "shots"
        goals = "goal" if tot["goals"] == 1 else "goals"
        fig.text(
            x_text,
            0.135,
            f"xG {tot['xg']:.2f}  |  {tot['shots']} regular {plural}, {tot['goals']} {goals}",
            color=INK,
            fontsize=10.5,
            ha=ha,
        )
    fig.text(
        0.5,
        0.188,
        "Match score (provider metadata)\n"
        f"{m.info.home_name} {m.info.home_score} - {m.info.away_score} {m.info.away_name}",
        fontsize=10,
        color=INK,
        ha="center",
        va="bottom",
    )
    _caption(
        fig,
        "Only regular shots are drawn: penalties, direct free-kick shots and own goals are "
        "excluded, so these totals need not equal the match score. xG is the sum of the "
        "model's shot probabilities (the selected location + body-part recipe, fitted on "
        "training matches only); it is a statement about shot quality, not about finishing "
        "or defending. Shot positions follow the provider's attacker-relative coordinates on "
        "an assumed 105 m x 68 m pitch. Attack directions are a display convention: the home "
        "team is drawn attacking right and the away team (positions rotated 180 degrees) "
        "attacking left, in both halves; the real directions of play switch at half-time. "
        f"One match from one season. {PAPER_NOTE} {ATTRIBUTION}",
        width=150,
    )
    return _save(fig, stem)


# --- building everything  ---------------------------------------------------------------------


def check_reproduces_validation(ev, reference: dict, tolerance: float = 1e-9) -> None:
    """The training-only models must give the validation scores that S04 recorded."""
    for model, recorded in reference.items():
        got = ev.scores["validation"][model]
        for metric in ("log_loss", "brier", "auc"):
            if abs(getattr(got, metric) - recorded[metric]) > tolerance:
                raise pdata.FigureInputError(
                    f"{model} validation {metric} {getattr(got, metric)!r} differs from the "
                    f"recorded S04 value {recorded[metric]!r}; the modelling code has changed"
                )
        if (got.n, got.goals) != (recorded["n"], recorded["goals"]):
            raise pdata.FigureInputError(f"{model} validation counts differ from S04")


def match_shots(ev, csv_path: Path, records: dict[int, dict], match_id: int, rows) -> MatchShots:
    """The chosen match's regular shots with the training-only model's probabilities."""
    validation = ev.parts["validation"]
    base = validation.base
    mask = base.match_ids == match_id
    if not mask.any():
        raise pdata.FigureInputError(f"match {match_id} is not in the validation window")
    if match_id not in records:
        raise pdata.FigureInputError(f"no provider match record for match {match_id}")
    info = pdata.parse_match_info(records[match_id])
    event_ids = base.event_ids[mask]
    teams = pdata.read_team_columns(csv_path, {int(e) for e in event_ids}, rows)
    team_id = np.array([teams[int(e)][0] for e in event_ids])
    side = np.array([teams[int(e)][1] for e in event_ids])
    expected_side = {info.home_id: "home", info.away_id: "away"}
    if set(team_id.tolist()) != set(expected_side):
        raise pdata.FigureInputError("shot rows and match metadata name different teams")
    for t, s in zip(team_id, side, strict=True):
        if expected_side[int(t)] != s:
            raise pdata.FigureInputError("a shot row's side disagrees with the match metadata")
    p = ev.predictions["validation"][NONLINEAR_LOCATION_BODY_PART][mask]
    y = base.y[mask]
    return MatchShots(
        info=info,
        competition=LEAGUE_NAMES[int(base.competition_ids[mask][0])],
        side=side,
        team_id=team_id,
        x_pct=base.x_pct[mask],
        y_pct=base.y_pct[mask],
        goal=y,
        xg=p,
        totals=pdata.team_totals(team_id, p, y),
    )


def build_figures(
    out_dir: Path = DEFAULT_OUT,
    csv_path: Path = DEFAULT_CSV,
    manifest_path: Path | None = None,
    summary_path: Path = pdata.S07_SUMMARY,
    log_path: Path = pdata.S07_LOG,
    validation_reference: Path = VALIDATION_SUMMARY,
    split_dir: Path = Path("data/processed"),
    records: list[dict] | None = None,
    rows=iter_csv_rows,
    locked=REAL_LOCKED,
    expected_match_id: int | None = RECORDED_MATCH_ID,
    reference_scores: dict | None = None,
    s03_report: Path = DEFAULT_S03_REPORT,
    protocol: Path = DEFAULT_PROTOCOL,
) -> dict:
    manifest_path = manifest_path or csv_path.with_suffix(".manifest.json")
    pdata.require_shot_table(csv_path)
    summary = pdata.load_frozen_summary(summary_path, log_path, locked)
    h = pdata.headline(summary)
    if reference_scores is None:
        if not validation_reference.exists():
            raise pdata.FigureInputError(f"{validation_reference} is missing")
        reference_scores = json.loads(validation_reference.read_text(encoding="utf-8"))[
            "validation_scores"
        ]
    if records is None:
        records = WyscoutSource(offline=True).matches()
    by_id = {int(r["wyId"]): r for r in records}

    # the unchanged S04 pipeline: train-only fit, guarded loaders, validation predictions
    ev = evaluate_nonlinear(csv_path, manifest_path, split_dir, 20, s03_report, protocol, rows)
    check_reproduces_validation(ev, reference_scores)
    candidate = ev.candidates[NONLINEAR_LOCATION_BODY_PART]
    if locked is REAL_LOCKED:
        if recipe_sha256(candidate.pipeline) != locked.primary_recipe_sha256:
            raise pdata.FigureInputError("the fitted pipeline differs from the locked recipe")
    train = ev.parts["train"]

    split_of = split_of_events(ev.identity, ev.plan)
    val_events = {int(e) for e, s in split_of.items() if s == "validation"}
    teams = pdata.read_team_columns(csv_path, val_events, rows)
    shots = [
        (int(m), str(d), teams[int(e)][0])
        for e, m, d in zip(
            ev.identity.event_ids, ev.identity.match_ids, ev.identity.dates, strict=True
        )
        if int(e) in val_events
    ]
    match_id = pdata.select_showcase_match(shots)
    if expected_match_id is not None and match_id != expected_match_id:
        raise pdata.FigureInputError(
            f"the selection rule now returns match {match_id}, not the recorded {expected_match_id}"
        )

    written = []
    written += plot_heldout_evidence(out_dir / STEMS[0], h)
    written += plot_shot_quality_map(
        out_dir / STEMS[1], probability_grid(candidate, train), len(train)
    )
    case = match_shots(ev, csv_path, by_id, match_id, rows)
    written += plot_match_shot_map(out_dir / STEMS[2], case)

    manifest = {
        "build_command": BUILD_COMMAND,
        "model": {
            "recipe": NONLINEAR_LOCATION_BODY_PART,
            "recipe_sha256": recipe_sha256(candidate.pipeline),
            "fitted_on": "original training split only",
            "training_shots": len(train),
        },
        "inputs": {
            "shot_table_sha256": ev.identity.csv_sha256,
            "split_assignment_sha256": ev.assignment_sha,
            "s07_summary_sha256": pdata.sha256_file(summary_path),
            "s07_protocol_sha256": summary["provenance"]["protocol_sha256"],
        },
        "case_study": {
            "match_id": match_id,
            "selection_rule": "docs/S08_VISUAL_SPEC.md section 5",
            "teams": [case.info.home_name, case.info.away_name],
            "regular_shot_totals": {
                str(t): {k: (round(v, 4) if k == "xg" else v) for k, v in tot.items()}
                for t, tot in case.totals.items()
            },
        },
        "png_dpi": PNG_DPI,
        "files": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": pdata.sha256_file(p)}
            for p in written
        ],
    }
    (out_dir / "asset_inventory.md").write_text(
        inventory_markdown(manifest, out_dir), encoding="utf-8"
    )
    return manifest


def inventory_markdown(manifest: dict, folder: Path) -> str:
    """A small human-readable list of the figure files, with sizes, pixels and hashes."""
    lines = [
        "# Portfolio figure inventory",
        "",
        f"Built with `{manifest['build_command']}`. Aggregates and one validation match only; "
        "no test-window shot-level data, no raw or derived shot tables.",
        "",
        "## Provenance",
        "",
        f"- Model: `{manifest['model']['recipe']}` (recipe SHA-256 "
        f"`{manifest['model']['recipe_sha256']}`), fitted on the {manifest['model']['fitted_on']} "
        f"({manifest['model']['training_shots']:,} shots). Figures 2 and 3 only.",
        f"- Shot table SHA-256 `{manifest['inputs']['shot_table_sha256']}`; split assignment "
        f"SHA-256 `{manifest['inputs']['split_assignment_sha256']}`.",
        f"- Figure 1 reads the frozen held-out summary, SHA-256 "
        f"`{manifest['inputs']['s07_summary_sha256']}` (protocol "
        f"`{manifest['inputs']['s07_protocol_sha256']}`); nothing is recomputed.",
        f"- Figure 3 case study: Wyscout match {manifest['case_study']['match_id']} "
        f"({' v '.join(manifest['case_study']['teams'])}), chosen by "
        f"{manifest['case_study']['selection_rule']}. Regular-shot totals by team id: "
        + "; ".join(
            f"{team}: {t['shots']} shots, {t['goals']} goal(s), xG {t['xg']:.2f}"
            for team, t in manifest["case_study"]["regular_shot_totals"].items()
        )
        + ".",
        "",
        "## Files",
        "",
        "| File | Bytes | Pixels | SHA-256 |",
        "|---|---|---|---|",
    ]
    for f in manifest["files"]:
        pixels = ""
        if f["path"].endswith(".png"):
            height, width = plt.imread(folder / f["path"]).shape[:2]
            pixels = f"{width} x {height}"
        lines.append(f"| `{f['path']}` | {f['bytes']:,} | {pixels} | `{f['sha256']}` |")
    lines += [
        "",
        "Data: Pappalardo & Massucco (2019), Wyscout open event data, CC BY 4.0, "
        "doi:10.6084/m9.figshare.c.4415000.v5; filtered, quality-screened and augmented for this "
        "project.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the S08 portfolio figures.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        manifest = build_figures(out_dir=args.out)
    except (pdata.FigureInputError, FileNotFoundError, DataAccessError) as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 1
    for f in manifest["files"]:
        print(f"{f['path']}  {f['bytes']:,} bytes")
    print("inventory: asset_inventory.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
