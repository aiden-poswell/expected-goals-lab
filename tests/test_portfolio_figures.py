"""S08: figure inputs, case-study selection, coordinates and the figure build.

Synthetic data only, except two read-only checks of the frozen S07 artefacts (skipped when the
local files are absent). No held-out feature or outcome is inspected: the build runs behind a
trap on every sensitive column of held-out-date rows (identity columns are still traversed, as
in S03-S05), and the final-evaluation code is booby-trapped.
"""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from expected_goals_lab import dataset as dataset_module
from expected_goals_lab import final_evaluation
from expected_goals_lab import portfolio_data as pdata
from expected_goals_lab import portfolio_figures as pf
from expected_goals_lab.evaluate_nonlinear import evaluate_nonlinear
from expected_goals_lab.final_lock import REAL_LOCKED
from tests import s04_helpers
from tests.s03_helpers import write_dataset
from tests.s04_helpers import GuardedRow, held_out_dates
from tests.s05_helpers import s05_rows

REPO = Path(__file__).resolve().parents[1]
SYNTH_PROTOCOL = "a" * 64
PRIMARY, BENCHMARK, CONSTANT = pdata.PRIMARY_KEY, pdata.BENCHMARK_KEY, pdata.CONSTANT_KEY


def synthetic_summary() -> dict:
    def score(ll, brier, auc, xg):
        return {"log_loss": ll, "brier": brier, "auc": auc, "expected_goals": xg,
                "goals": 700, "shots": 6000}  # fmt: skip

    return {
        "provenance": {"protocol_sha256": SYNTH_PROTOCOL},
        "test": {
            "shots": 6000,
            "goals": 700,
            "matches": 280,
            "first_date": "2018-04-14",
            "last_date": "2018-05-20",
            "dates": 27,
        },  # fmt: skip
        "scores": {
            PRIMARY: score(0.29, 0.084, 0.79, 690.0),
            BENCHMARK: score(0.30, 0.087, 0.77, 688.0),
            CONSTANT: score(0.35, 0.099, 0.5, 670.0),
        },
        "intervals": {
            "log_loss": {"estimate": -0.01, "low": -0.013, "high": -0.007},
            "brier": {"estimate": -0.003, "low": -0.004, "high": -0.002},
        },
    }


def write_s07(folder: Path, summary: dict | None = None) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    summary_path, log_path = folder / "summary.json", folder / "log.jsonl"
    summary_path.write_text(json.dumps(summary or synthetic_summary()), encoding="utf-8")
    sha = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    entries = [{"outcome": "started"}, {"outcome": "completed", "summary_sha256": sha}]
    log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return summary_path, log_path


def records_for(rows, swap_sides: bool = False) -> list[dict]:
    """Provider-style match records for the synthetic matches (team 10 home, team 11 away)."""
    home, away = (11, 10) if swap_sides else (10, 11)
    seen = {}
    for r in rows:
        seen.setdefault(r.match_id, r.match_datetime_utc)
    return [
        {
            "wyId": match_id,
            "label": "Home FC - Away FC, 2 - 1",
            "dateutc": date,
            "teamsData": {
                str(home): {"side": "home", "score": 2},
                str(away): {"side": "away", "score": 1},
            },
        }
        for match_id, date in seen.items()
    ]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    folder = tmp_path_factory.mktemp("s08")
    csv_path, manifest_path, rows = write_dataset(folder / "data", s05_rows())
    s07_summary, s07_log = write_s07(folder / "s07")
    no_report = folder / "absent.md"
    ev = evaluate_nonlinear(csv_path, manifest_path, folder / "split", 10, no_report, no_report)
    reference = {
        m: {"log_loss": s.log_loss, "brier": s.brier, "auc": s.auc, "n": s.n, "goals": s.goals}
        for m, s in ev.scores["validation"].items()
    }
    locked = REAL_LOCKED.__class__(**{**REAL_LOCKED.__dict__, "protocol_sha256": SYNTH_PROTOCOL})
    return SimpleNamespace(
        folder=folder, csv=csv_path, manifest=manifest_path, rows=rows, ev=ev, reference=reference,
        s07_summary=s07_summary, s07_log=s07_log, no_report=no_report, locked=locked,
        test_dates=held_out_dates(csv_path, manifest_path),
    )  # fmt: skip


def build(synth, out, rows=dataset_module.iter_csv_rows, **overrides):
    kwargs = dict(
        out_dir=out, csv_path=synth.csv, manifest_path=synth.manifest,
        summary_path=synth.s07_summary, log_path=synth.s07_log, split_dir=synth.folder / "split",
        records=records_for(synth.rows), rows=rows, locked=synth.locked, expected_match_id=None,
        reference_scores=synth.reference, s03_report=synth.no_report, protocol=synth.no_report,
    )  # fmt: skip
    kwargs.update(overrides)
    return pf.build_figures(**kwargs)


# --- Figure 1 inputs: the frozen S07 summary --------------------------------------------


def test_the_frozen_summary_loads_and_headline_is_arithmetic_on_it(tmp_path):
    summary_path, log_path = write_s07(tmp_path)
    locked = REAL_LOCKED.__class__(**{**REAL_LOCKED.__dict__, "protocol_sha256": SYNTH_PROTOCOL})
    summary = pdata.load_frozen_summary(summary_path, log_path, locked)
    h = pdata.headline(summary)
    assert (h["shots"], h["goals"], h["matches"]) == (6000, 700, 280)
    assert h["relative_log_loss"] == pytest.approx((0.29 - 0.30) / 0.30)
    assert h["relative_brier"] == pytest.approx((0.084 - 0.087) / 0.087)
    assert h["auc_gain"] == pytest.approx(0.02)
    assert h["intervals"] == summary["intervals"]


def test_missing_altered_or_foreign_s07_inputs_fail_clearly(tmp_path):
    locked = REAL_LOCKED.__class__(**{**REAL_LOCKED.__dict__, "protocol_sha256": SYNTH_PROTOCOL})
    summary_path, log_path = write_s07(tmp_path)
    with pytest.raises(pdata.FigureInputError, match="missing"):
        pdata.load_frozen_summary(tmp_path / "nope.json", log_path, locked)
    with pytest.raises(pdata.FigureInputError, match="missing"):
        pdata.load_frozen_summary(summary_path, tmp_path / "nope.jsonl", locked)
    original = summary_path.read_bytes()
    summary_path.write_bytes(original + b" ")
    with pytest.raises(pdata.FigureInputError, match="hash"):
        pdata.load_frozen_summary(summary_path, log_path, locked)
    summary_path.write_bytes(original)
    other = REAL_LOCKED.__class__(**{**REAL_LOCKED.__dict__, "protocol_sha256": "b" * 64})
    with pytest.raises(pdata.FigureInputError, match="locked protocol"):
        pdata.load_frozen_summary(summary_path, log_path, other)
    log_path.write_text(json.dumps({"outcome": "started"}) + "\n", encoding="utf-8")
    with pytest.raises(pdata.FigureInputError, match="exactly one completed"):
        pdata.load_frozen_summary(summary_path, log_path, locked)


def test_a_summary_without_the_expected_scores_is_refused(tmp_path):
    locked = REAL_LOCKED.__class__(**{**REAL_LOCKED.__dict__, "protocol_sha256": SYNTH_PROTOCOL})
    summary = synthetic_summary()
    del summary["scores"][CONSTANT]
    summary_path, log_path = write_s07(tmp_path, summary)
    with pytest.raises(pdata.FigureInputError, match="lacks"):
        pdata.load_frozen_summary(summary_path, log_path, locked)


REAL_S07 = REPO / "data/processed/final_test_summary.json"


@pytest.mark.skipif(not REAL_S07.exists(), reason="local S07 artefacts not present")
def test_real_s07_summary_is_read_without_being_touched():
    before = {p: pdata.sha256_file(p) for p in (REAL_S07, REPO / pdata.S07_LOG)}
    summary = pdata.load_frozen_summary(REAL_S07, REPO / pdata.S07_LOG)
    h = pdata.headline(summary)
    assert (h["shots"], h["goals"], h["matches"]) == (6653, 738, 298)
    assert h["scores"][PRIMARY]["log_loss"] == pytest.approx(0.290475, abs=1e-6)
    assert h["scores"][BENCHMARK]["brier"] == pytest.approx(0.087518, abs=1e-6)
    assert h["scores"][PRIMARY]["auc"] == pytest.approx(0.786533, abs=1e-6)
    assert h["relative_log_loss"] == pytest.approx(-0.0311, abs=5e-4)
    assert before == {p: pdata.sha256_file(p) for p in before}
    assert final_evaluation.REAL_EXECUTION_ENABLED is False


# --- coordinates and pitch --------------------------------------------------------------


def test_display_coordinates_for_both_sides():
    x, y = pdata.to_display(np.array([100.0, 50.0]), np.array([50.0, 0.0]), "home")
    assert x.tolist() == [105.0, 52.5] and y.tolist() == [34.0, 0.0]
    x, y = pdata.to_display(np.array([100.0, 90.0]), np.array([50.0, 30.0]), "away")
    assert x[0] == pytest.approx(0.0) and y[0] == pytest.approx(34.0)  # away goal attacked: left
    assert x[1] == pytest.approx(10.5) and y[1] == pytest.approx(47.6)
    with pytest.raises(pdata.FigureInputError):
        pdata.to_display(np.array([50.0]), np.array([50.0]), "neutral")


def test_the_same_attacker_relative_shot_lands_at_opposite_ends():
    hx, hy = pdata.to_display(np.array([88.0]), np.array([40.0]), "home")
    ax_, ay = pdata.to_display(np.array([88.0]), np.array([40.0]), "away")
    assert hx[0] + ax_[0] == pytest.approx(105.0) and hy[0] + ay[0] == pytest.approx(68.0)


def test_pitch_markings_follow_the_laws_of_the_game():
    from matplotlib.patches import Rectangle

    m = pf.pitch_markings()
    assert m["penalty_area"] == (16.5, 40.32) and m["six_yard_box"] == (5.5, 18.32)
    assert m["penalty_spot"] == 11.0 and m["circle_radius"] == 9.15 and m["goal_width"] == 7.32
    fig, ax = pf.plt.subplots()
    pf.draw_pitch(ax)
    sizes = {(round(p.get_width(), 2), round(p.get_height(), 2)) for p in ax.patches
             if isinstance(p, Rectangle)}  # fmt: skip
    assert {(105.0, 68.0), (16.5, 40.32), (5.5, 18.32), (2.0, 7.32)} <= sizes
    assert ax.get_ylim()[0] > ax.get_ylim()[1], "y grows downwards, as in the provider's data"
    pf.plt.close(fig)


# --- case-study selection ---------------------------------------------------------------


def shots_of(match_id, date, teams):
    return [(match_id, date, t) for t, n in teams.items() for _ in range(n)]


def test_selection_takes_the_earliest_qualifying_match_then_lowest_id():
    shots = (
        shots_of(9, "2018-03-01", {1: 5, 2: 5})
        + shots_of(8, "2018-03-02", {1: 6, 2: 6})
        + shots_of(7, "2018-03-02", {1: 6, 2: 6})
    )
    assert pdata.select_showcase_match(shots) == 9
    assert pdata.select_showcase_match(shots[::-1]) == 9, "input order must not matter"
    assert pdata.select_showcase_match([s for s in shots if s[0] != 9]) == 7  # id tie-break


def test_selection_requires_both_teams_and_ten_shots_and_ignores_early_failures():
    early_one_team = shots_of(1, "2018-02-26", {1: 12})
    early_short = shots_of(2, "2018-02-26", {1: 5, 2: 4})
    qualifying = shots_of(3, "2018-02-27", {1: 5, 2: 5})
    assert pdata.select_showcase_match(early_one_team + early_short + qualifying) == 3
    with pytest.raises(pdata.FigureInputError, match="no validation match"):
        pdata.select_showcase_match(early_one_team + early_short)
    with pytest.raises(pdata.FigureInputError, match="more than one date"):
        pdata.select_showcase_match([(5, "2018-03-01", 1), (5, "2018-03-02", 2)])


def test_selection_uses_no_outcome_columns():
    import inspect

    assert list(inspect.signature(pdata.select_showcase_match).parameters) == ["shots"]
    source = inspect.getsource(pdata.select_showcase_match)
    for word in ("is_goal", "xg", "proba", ".y", "label"):
        assert word not in source, word


def test_selection_is_unchanged_when_every_label_is_flipped(synth, tmp_path):
    import dataclasses

    flipped = [dataclasses.replace(r, is_goal=1 - r.is_goal) for r in synth.rows]
    csv2, man2, _ = write_dataset(tmp_path / "flipped", flipped)

    def chosen(csv_path, manifest_path):
        ev = evaluate_nonlinear(csv_path, manifest_path, tmp_path / "s", 5, synth.no_report,
                                synth.no_report)  # fmt: skip
        split_of = pf.split_of_events(ev.identity, ev.plan)
        val = {int(e) for e, s in split_of.items() if s == "validation"}
        teams = pdata.read_team_columns(csv_path, val)
        shots = [(int(m), str(d), teams[int(e)][0]) for e, m, d in
                 zip(ev.identity.event_ids, ev.identity.match_ids, ev.identity.dates, strict=True)
                 if int(e) in val]  # fmt: skip
        return pdata.select_showcase_match(shots)

    assert chosen(synth.csv, synth.manifest) == chosen(csv2, man2)


# --- match metadata ---------------------------------------------------------------------


def record(label="A FC - B FC, 2 - 1", home=(10, 2), away=(11, 1)):
    return {
        "wyId": 77,
        "label": label,
        "dateutc": "2018-03-01 15:00:00",
        "teamsData": {
            str(home[0]): {"side": "home", "score": home[1]},
            str(away[0]): {"side": "away", "score": away[1]},
        },
    }


def test_match_info_names_sides_and_score_come_from_the_record():
    info = pdata.parse_match_info(record())
    assert (info.home_name, info.away_name) == ("A FC", "B FC")
    assert (info.home_id, info.away_id, info.home_score, info.away_score) == (10, 11, 2, 1)


@pytest.mark.parametrize(
    "bad",
    [
        record(label="A FC vs B FC"),
        record(label="A FC - B FC, 1 - 2"),  # label order disagrees with the per-team scores
        {**record(), "teamsData": {"10": {"side": "home", "score": 2}}},
        {
            **record(),
            "teamsData": {"10": {"side": "home", "score": 2}, "11": {"side": "home", "score": 1}},
        },
    ],
)
def test_inconsistent_match_metadata_stops_the_figure(bad):
    with pytest.raises(pdata.FigureInputError):
        pdata.parse_match_info(bad)


def test_team_totals_are_per_team_sums():
    team = np.array([10, 11, 10, 11, 11])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([0, 1, 1, 0, 0])
    totals = pdata.team_totals(team, p, y)
    assert totals[10] == {"shots": 2, "goals": 1, "xg": pytest.approx(0.4)}
    assert totals[11] == {"shots": 3, "goals": 1, "xg": pytest.approx(1.1)}
    assert sum(t["xg"] for t in totals.values()) == pytest.approx(p.sum())


def test_support_counts_on_a_handmade_grid():
    xs, ys = np.array([10.0, 10.0, 11.0, 30.0]), np.array([5.0, 6.0, 5.0, 5.0])
    counts = pdata.support_counts(xs, ys, np.array([10.5, 30.0, 60.0]), np.array([5.5]), 3.0)
    assert counts.tolist() == [[3, 1, 0]]


# --- the build --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def built(synth, tmp_path_factory):
    """One synthetic build behind the trap, with the final-evaluation code booby-trapped."""
    out = tmp_path_factory.mktemp("figs")
    trap: list[str] = []
    mp = pytest.MonkeyPatch()
    mp.setattr(s04_helpers, "SENSITIVE", s04_helpers.SENSITIVE | {"team_id", "side", "player_id"})

    def explode(*args, **kwargs):
        raise AssertionError("final-test code was called by the figure build")

    for name in ("execute_final_evaluation", "load_final_test_partition",
                 "load_final_test_body_parts", "main"):  # fmt: skip
        mp.setattr(final_evaluation, name, explode)
    mp.setattr(dataset_module, "load_final_test_partition", explode)

    def reader(path):
        for row in dataset_module.iter_csv_rows(path):
            yield GuardedRow(row, row["match_datetime_utc"][:10] in synth.test_dates, trap)

    manifest = build(synth, out, rows=reader)
    mp.undo()
    return SimpleNamespace(out=out, manifest=manifest, trap=trap)


def test_the_build_never_touches_a_held_out_row_or_the_final_evaluation(built):
    assert built.trap == []


def test_the_build_writes_three_figures_in_two_formats_and_a_manifest(built):
    names = sorted(p.name for p in built.out.iterdir())
    for stem in pf.STEMS:
        assert f"{stem}.png" in names and f"{stem}.svg" in names
    assert "asset_inventory.md" in names and "figure_manifest.json" not in names
    assert not any(n.endswith((".csv", ".zip", ".pkl", ".parquet")) for n in names)
    assert all(f["bytes"] > 1000 for f in built.manifest["files"])
    assert all(
        hashlib.sha256((built.out / f["path"]).read_bytes()).hexdigest() == f["sha256"]
        for f in built.manifest["files"]
    )


def test_the_inventory_lists_every_figure_with_its_size(built):
    text = (built.out / "asset_inventory.md").read_text(encoding="utf-8")
    for f in built.manifest["files"]:
        assert f"`{f['path']}` | {f['bytes']:,} |" in text
        assert (built.out / f["path"]).stat().st_size == f["bytes"]
    assert "CC BY 4.0" in text and "no raw or derived shot tables" in text


def test_png_resolution_is_at_least_160_dpi(built):
    from PIL import Image

    assert pf.PNG_DPI >= 160
    image = Image.open(built.out / f"{pf.STEMS[2]}.png")
    assert image.width >= 160 * 6


def test_manifest_records_model_split_data_and_command(built, synth):
    m = built.manifest
    assert m["build_command"] == "uv run python -m expected_goals_lab.portfolio_figures"
    assert m["model"]["recipe"] == "nonlinear_location_body_part"
    assert m["model"]["fitted_on"] == "original training split only"
    assert m["model"]["training_shots"] == len(synth.ev.parts["train"])
    assert m["inputs"]["shot_table_sha256"] == synth.ev.identity.csv_sha256
    assert m["inputs"]["split_assignment_sha256"] == synth.ev.assignment_sha
    assert m["inputs"]["s07_summary_sha256"] == pdata.sha256_file(synth.s07_summary)


def test_the_case_study_match_is_the_independently_expected_one(built, synth):
    val_dates = {d for d, s in synth.ev.plan.date_split.items() if s == "validation"}
    per = {}
    for r in synth.rows:
        if r.eligible and r.match_datetime_utc[:10] in val_dates:
            per.setdefault((r.match_datetime_utc[:10], r.match_id), set()).add(r.team_id)
    counts = {}
    for r in synth.rows:
        if r.eligible and r.match_datetime_utc[:10] in val_dates:
            counts[r.match_id] = counts.get(r.match_id, 0) + 1
    expected = min(k for k, teams in per.items() if len(teams) == 2 and counts[k[1]] >= 10)[1]
    assert built.manifest["case_study"]["match_id"] == expected
    assert built.manifest["case_study"]["teams"] == ["Home FC", "Away FC"]


def test_case_totals_match_independent_sums_and_exclude_quarantined_rows(built, synth):
    match_id = built.manifest["case_study"]["match_id"]
    mine = [r for r in synth.rows if r.match_id == match_id]
    assert any(not r.eligible for r in mine), "the fixture holds quarantined rows to exclude"
    totals = built.manifest["case_study"]["regular_shot_totals"]
    for team in (10, 11):
        eligible = [r for r in mine if r.eligible and r.team_id == team]
        assert totals[str(team)]["shots"] == len(eligible)
        assert totals[str(team)]["goals"] == sum(r.is_goal for r in eligible)
    assert sum(t["shots"] for t in totals.values()) == sum(r.eligible for r in mine)
    assert all(0 < t["xg"] < t["shots"] for t in totals.values())


def test_the_build_is_deterministic(built, synth, tmp_path):
    again = build(synth, tmp_path / "again")
    for f in built.manifest["files"]:
        assert (tmp_path / "again" / f["path"]).read_bytes() == (built.out / f["path"]).read_bytes()
    assert again["case_study"] == built.manifest["case_study"]


def test_an_altered_training_model_stops_the_build(synth, tmp_path):
    wrong = copy.deepcopy(synth.reference)
    wrong[PRIMARY]["log_loss"] += 1e-6
    with pytest.raises(pdata.FigureInputError, match="differs from the recorded S04 value"):
        build(synth, tmp_path / "x", reference_scores=wrong)
    assert not (tmp_path / "x").exists()


def test_a_changed_selection_rule_result_stops_the_build(synth, tmp_path):
    with pytest.raises(pdata.FigureInputError, match="not the recorded"):
        build(synth, tmp_path / "y", expected_match_id=1)


def test_swapped_sides_in_the_match_metadata_stop_the_build(synth, tmp_path):
    with pytest.raises(pdata.FigureInputError, match="disagrees with the match metadata"):
        build(synth, tmp_path / "z", records=records_for(synth.rows, swap_sides=True))


def test_missing_match_record_or_inputs_fail_clearly(synth, tmp_path):
    with pytest.raises(pdata.FigureInputError, match="no provider match record"):
        build(synth, tmp_path / "a", records=[])
    with pytest.raises(pdata.FigureInputError, match="missing"):
        build(synth, tmp_path / "b", summary_path=tmp_path / "gone.json")
    with pytest.raises(pdata.FigureInputError, match="missing"):
        build(
            synth, tmp_path / "c", reference_scores=None, validation_reference=tmp_path / "x.json"
        )


def test_the_cli_reports_missing_inputs_without_a_traceback(tmp_path, capsys):
    assert pf.main(["--out", str(tmp_path / "o")]) in (0, 1)  # real inputs or a clear STOP
    assert "Traceback" not in capsys.readouterr().err


# --- the model grid and the drawn figures -----------------------------------------------


def test_probability_grid_is_blank_without_training_support_and_valid_elsewhere(synth, monkeypatch):
    monkeypatch.setattr(pdata, "SUPPORT_MIN_SHOTS", 3)  # the small synthetic training split is thin
    candidate = synth.ev.candidates[pf.NONLINEAR_LOCATION_BODY_PART]
    grid = pf.probability_grid(candidate, synth.ev.parts["train"])
    assert (
        grid.x_m.min() > 60 and grid.x_m.max() < 105 and grid.y_m.min() > 0 and grid.y_m.max() < 68
    )
    for cls, p in grid.probability.items():
        supported = grid.support[cls] >= pdata.SUPPORT_MIN_SHOTS
        assert np.array_equal(np.isnan(p), ~supported)
        assert np.all((p[supported] >= 0) & (p[supported] <= 1))
    assert any(np.isfinite(p).any() for p in grid.probability.values())


def test_grid_estimates_equal_the_models_own_prediction_at_a_location(synth):
    from expected_goals_lab.geometry import distance_to_goal, goal_angle

    candidate = synth.ev.candidates[pf.NONLINEAR_LOCATION_BODY_PART]
    x_m, y_m = pf.grid_axes()
    part = pf._grid_partition(x_m[-3:], y_m[30:32], "right_foot")
    p = candidate.predict_proba(part)
    assert len(p) == 6
    assert part.base.features["distance_m"][0] == pytest.approx(distance_to_goal(x_m[-3], y_m[30]))
    assert part.base.features["angle_rad"][0] == pytest.approx(goal_angle(x_m[-3], y_m[30]))


@pytest.fixture
def captured(monkeypatch):
    figures = []
    monkeypatch.setattr(pf, "_save", lambda fig, stem: figures.append(fig) or [])
    return figures


def texts(fig):
    return " ".join(t.get_text() for t in fig.findobj(lambda o: hasattr(o, "get_text")))


def test_every_figure_carries_attribution_and_licence(captured, synth):
    h = pdata.headline(synthetic_summary())
    pf.plot_heldout_evidence(Path("a"), h)
    candidate = synth.ev.candidates[pf.NONLINEAR_LOCATION_BODY_PART]
    pf.plot_shot_quality_map(
        Path("b"), pf.probability_grid(candidate, synth.ev.parts["train"]), 1234
    )
    assert len(captured) == 2
    for fig in captured:
        text = texts(fig)
        for needle in ("CC BY 4.0", "10.6084/m9.figshare.c.4415000.v5", "Pappalardo", "Wyscout"):
            assert needle in text
    assert "Head / other body" in texts(captured[1]) and "header" in texts(captured[1])
    assert "Headers" not in texts(captured[1])


def test_figure_one_prints_the_summary_values_and_the_zero_axis_disclosure(captured):
    h = pdata.headline(synthetic_summary())
    pf.plot_heldout_evidence(Path("a"), h)
    text = texts(captured[0])
    needles = ["0.2900", "0.3000", "0.35", "0.08400", "0.08700", "−0.0100", "−0.0130", "6,000"]
    needles += ["280", "not at zero", "bars start at zero", "700", "690.0"]
    for needle in needles:
        assert needle in text, needle


def test_case_map_marks_scale_with_xg_and_goals_are_filled(captured, synth):
    ev = synth.ev
    validation = ev.parts["validation"].base
    match_id = int(validation.match_ids[0])
    shots = pf.match_shots(
        ev, synth.csv, {int(r["wyId"]): r for r in records_for(synth.rows)}, match_id,
        dataset_module.iter_csv_rows,
    )  # fmt: skip
    pf.plot_match_shot_map(Path("m"), shots)
    fig = captured[0]
    pitch = fig.axes[0]
    shots_drawn = [c for c in pitch.collections if c.get_zorder() in (3, 4)]
    dots = [c for c in pitch.collections if c.get_zorder() == 6]
    drawn = np.round(np.vstack([c.get_offsets() for c in dots]), 6)
    expected = []
    for side in ("home", "away"):
        sel = shots.side == side
        px, py = pdata.to_display(shots.x_pct[sel], shots.y_pct[sel], side)
        expected += list(zip(np.round(px, 6), np.round(py, 6), strict=True))
    assert sorted(map(tuple, drawn)) == sorted(expected), "a dot marks every recorded origin"
    assert all(np.all(c.get_sizes() == pf.ORIGIN_DOT_AREA) for c in dots)
    sizes = np.concatenate([np.ravel(c.get_sizes()) for c in shots_drawn])
    assert sorted(np.round(sizes, 6)) == sorted(np.round(pf.XG_AREA_PER_UNIT * shots.xg, 6))
    goals_drawn = sum(
        len(c.get_sizes()) for c in shots_drawn if c.get_edgecolor().shape[0] and
        tuple(np.round(c.get_edgecolor()[0][:3], 2)) == (0.07, 0.07, 0.07)
    )  # fmt: skip
    assert goals_drawn == int(shots.goal.sum())
    assert 0 < shots.xg.max() < 1 and set(np.unique(shots.goal)) <= {0, 1}
    assert shots.totals[10]["shots"] + shots.totals[11]["shots"] == len(shots.xg)
    assert "Home FC" in texts(fig) and "need not equal the match score" in texts(fig)
    assert "penalties, direct free-kick shots and own goals" in texts(fig)


# --- S08A wording corrections ------------------------------------------------------------------


def test_story_and_spec_make_only_the_claims_the_evidence_supports():
    story = (REPO / "reports/football_research_story.md").read_text(encoding="utf-8")
    spec = (REPO / "docs/S08_VISUAL_SPEC.md").read_text(encoding="utf-8")
    assert "not an ablation of body part alone" in story
    assert "validation window only and was not part of the held-out evaluation" in story
    assert "display convention" in story and "not the real direction of play" in story
    assert "the held-out data therefore do not separate the two ingredients" in story
    assert "No final-test row is read, scored, plotted" not in spec
    assert "identity columns" in spec and "traversed in full" in spec


def test_figure_text_states_direction_convention_and_the_comparison_limit(captured, synth):
    h = pdata.headline(synthetic_summary())
    pf.plot_heldout_evidence(Path("a"), h)
    assert "not an ablation of body part alone" in texts(captured[0])
    validation = synth.ev.parts["validation"].base
    shots = pf.match_shots(
        synth.ev, synth.csv, {int(r["wyId"]): r for r in records_for(synth.rows)},
        int(validation.match_ids[0]), dataset_module.iter_csv_rows,
    )  # fmt: skip
    pf.plot_match_shot_map(Path("m"), shots)
    text = texts(captured[1])
    assert "display convention" in text and "real directions of play switch at half-time" in text
    assert "small black dot = recorded shot position" in text
