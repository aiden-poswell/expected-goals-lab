"""S08A: the shot-origin audit, on a synthetic event archive and a synthetic shot table.

No provider data. The audit runs behind traps: an event of a held-out match raises if any field
other than ``matchId`` is read, and every sensitive column of a held-out-date shot-table row raises.
Two checks read the committed audit report as text and skip when it is absent.
"""

import dataclasses
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from expected_goals_lab import portfolio_data as pdata
from expected_goals_lab import portfolio_figures as pf
from expected_goals_lab import shot_origin_audit as audit
from expected_goals_lab.dataset import iter_csv_rows
from expected_goals_lab.evaluate_nonlinear import evaluate_nonlinear
from expected_goals_lab.geometry import shot_geometry
from expected_goals_lab.wyscout_shots import _start_position, parse_shot
from tests import s04_helpers
from tests.s03_helpers import write_dataset
from tests.s04_helpers import GuardedRow, held_out_dates
from tests.s05_helpers import s05_rows

REPO = Path(__file__).resolve().parents[1]
BODY_TAG = {"left_foot": 401, "right_foot": 402, "head_or_body": 403}


# --- the production parser picks the start entry only ---------------------------------


def match_record(match_id=1, home=10, away=11):
    return {
        "wyId": match_id,
        "competitionId": 364,
        "seasonId": 1,
        "teamsData": {
            str(home): {"side": "home", "teamId": home},
            str(away): {"side": "away", "teamId": away},
        },
    }


def shot_event(positions, tags=(402,), match_id=1, team=10, event_id=99):
    return {
        "id": event_id,
        "matchId": match_id,
        "teamId": team,
        "playerId": 5,
        "eventId": 10,
        "subEventId": 100,
        "matchPeriod": "1H",
        "eventSec": 12.5,
        "positions": positions,
        "tags": [{"id": t} for t in tags],
    }


def test_the_parser_uses_the_first_positions_entry_and_ignores_the_second():
    event = shot_event([{"x": 97, "y": 54}, {"x": 3, "y": 11}])
    shot = parse_shot(event, match_record())
    assert (shot.x, shot.y) == (97, 54)
    swapped = shot_event([{"x": 3, "y": 11}, {"x": 97, "y": 54}])
    shot = parse_shot(swapped, match_record())
    assert (shot.x, shot.y) == (3, 11), "the second entry is never promoted to the origin"


def test_goal_mouth_and_goal_tags_do_not_change_the_origin():
    positions = [{"x": 90, "y": 30}, {"x": 100, "y": 100}]
    plain = parse_shot(shot_event(positions, tags=(402,)), match_record())
    zoned = parse_shot(shot_event(positions, tags=(101, 402, 1201, 1223)), match_record())
    assert (plain.x, plain.y) == (zoned.x, zoned.y) == (90, 30)
    assert zoned.is_goal and not plain.is_goal


def test_an_unusable_first_entry_gives_no_origin_instead_of_a_substitute():
    for positions in (
        [],
        [{"x": "a", "y": 1}, {"x": 50, "y": 50}],
        [{"x": 120, "y": 5}, {"x": 5, "y": 5}],
        None,
    ):
        assert _start_position(positions) == (None, None)


def test_second_entry_classification_and_goal_mouth_tags():
    assert audit.second_position_class([{"x": 1, "y": 1}]) == "absent"
    assert audit.second_position_class([{"x": 1, "y": 1}, {"x": 0, "y": 0}]) == "(0, 0) placeholder"
    placeholder = audit.second_position_class([{"x": 1, "y": 1}, {"x": 100, "y": 100}])
    assert placeholder == "(100, 100) placeholder"
    assert audit.second_position_class([{"x": 1, "y": 1}, {"x": 40, "y": 60}]) == "other"
    assert audit.goal_mouth_tags(shot_event([], tags=(101, 1203, 402, 1223, 1224))) == [1203, 1223]


# --- origin geometry ------------------------------------------------------------------


def test_origin_geometry_is_reproducible_by_hand():
    g = shot_geometry(97, 54)
    x_m, y_m = 105 * 0.97, 68 * 0.54
    assert (g.x_m, g.y_m) == (pytest.approx(x_m), pytest.approx(y_m))
    assert g.distance_m == pytest.approx(math.hypot(105 - x_m, 34 - y_m))
    a = np.array([105 - x_m, 34 - 3.66 - y_m])
    b = np.array([105 - x_m, 34 + 3.66 - y_m])
    expected = math.acos(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    assert g.angle_rad == pytest.approx(expected)
    assert round(g.distance_m, 2) == 4.16 and round(g.angle_deg, 1) == 80.3


def test_goal_line_and_centre_extremes():
    on_line = shot_geometry(100, 50)
    assert on_line.distance_m == pytest.approx(0.0) and on_line.angle_deg == pytest.approx(180.0)
    outside_posts = shot_geometry(100, 70)
    assert outside_posts.angle_rad == pytest.approx(0.0)
    far_goal = shot_geometry(60, 50)
    assert far_goal.distance_m == pytest.approx(42.0)
    assert shot_geometry(97, 54).angle_rad == pytest.approx(shot_geometry(97, 46).angle_rad)


# --- bands, flags, tables -------------------------------------------------------------


@pytest.mark.parametrize(
    "x,band",
    [(100, "x = 100"), (99, "x = 99"), (98, "x = 98"), (97, "x = 95-97"), (95, "x = 95-97"),
     (94, "x = 90-94"), (90, "x = 90-94"), (89, "x < 90"), (20, "x < 90")],
)  # fmt: skip
def test_x_bands_are_mutually_exclusive_and_cover_the_threshold_edges(x, band):
    assert audit.x_band(x) == band
    assert sum(audit.x_band(x) == b for b in audit.BANDS) == 1


def test_diagnostic_flags_are_separate_thresholds():
    assert audit.diagnostic_flags(94, 50) == {
        "x>=95": False,
        "x>=98": False,
        "x=100": False,
        "(100,50)": False,
    }
    assert audit.diagnostic_flags(95, 50)["x>=95"] and not audit.diagnostic_flags(95, 50)["x>=98"]
    assert audit.diagnostic_flags(98, 10)["x>=98"] and not audit.diagnostic_flags(98, 10)["x=100"]
    assert (
        audit.diagnostic_flags(100, 10)["x=100"] and not audit.diagnostic_flags(100, 10)["(100,50)"]
    )
    assert audit.diagnostic_flags(100, 50)["(100,50)"]


def test_centre_zones():
    assert audit.centre_zone(100, 50) == "(100, 50) exactly"
    assert audit.centre_zone(100, 47) == "x = 100, other y inside the goal mouth"
    assert audit.centre_zone(100, 70) == "x = 100, y outside"
    assert audit.centre_zone(96, 55) == "x = 95-99, y inside the goal mouth"
    assert audit.centre_zone(96, 56) == "x = 95-99, y outside"
    assert audit.centre_zone(80, 50) == "x < 95"


def test_band_and_zone_tables_reconcile_goal_and_non_goal_counts():
    rng = np.random.default_rng(1)
    x = rng.integers(60, 101, 400)
    y = rng.integers(0, 101, 400)
    goal = (rng.random(400) < 0.2).astype(int)
    for rows in (audit.band_table(x, goal), audit.zone_table(x, y, goal)):
        assert sum(r[1] for r in rows) == 400
        assert sum(r[2] for r in rows) == int(goal.sum())
        assert all(r[1] == r[2] + r[3] for r in rows)
    sub = audit.subset_of_concern(x, goal)
    assert sub["rows"] == int((x == 100).sum()) and sub["goals"] + sub["non_goals"] == sub["rows"]


def test_empty_band_prints_na_not_a_division_error():
    rows = audit.band_table(np.array([60, 61]), np.array([0, 1]))
    assert rows[0][1:] == [0, 0, 0, "n/a"]


def test_the_mechanical_screen_follows_the_predeclared_criteria():
    clean = {"rows": 3, "share": 0.0005, "goals": 1, "non_goals": 2}
    assert audit.mechanical_screen(0, 0.7, clean)[0] == "PASS"
    assert (
        audit.mechanical_screen(1, 0.7, clean)[0] == "FAIL"
    )  # an origin differs from positions[0]
    assert audit.mechanical_screen(0, 0.7, {**clean, "share": 0.006})[0] == "FAIL"
    assert audit.mechanical_screen(0, 0.7, {**clean, "share": 0.003})[0] == "CONDITIONAL PASS"
    all_goals = {**clean, "goals": 3, "non_goals": 0}
    assert audit.mechanical_screen(0, 0.7, all_goals)[0] == "CONDITIONAL PASS"
    assert audit.mechanical_screen(0, 0.2, clean)[0] == "CONDITIONAL PASS"
    empty = {"rows": 0, "share": 0.0, "goals": 0, "non_goals": 0}
    assert audit.mechanical_screen(0, 0.7, empty)[0] == "PASS"


def test_the_mechanical_screen_takes_no_destination_flag():
    """The old literal ``False`` argument is gone: the screen cannot claim to have checked it."""
    import inspect

    params = list(inspect.signature(audit.mechanical_screen).parameters)
    assert params == ["mismatches", "near_goal_non_goal_share", "subset"]
    assert not hasattr(audit, "verdict")
    assert "destination_evidence" not in inspect.getsource(audit.run_audit)


def review(conclusion):
    return audit.ManualReview(conclusion, "2026-10-22", ("basis",))


def test_a_positive_finding_needs_both_a_clean_screen_and_a_manual_review():
    assert audit.overall_finding("PASS", review(audit.REVIEW_NO_EVIDENCE))[0] == "PASS"
    assert (
        audit.overall_finding("CONDITIONAL PASS", review(audit.REVIEW_NO_EVIDENCE))[0]
        == "CONDITIONAL PASS"
    )
    finding, basis = audit.overall_finding("PASS", review(audit.REVIEW_NO_EVIDENCE))
    assert "review judgement" in basis and "mechanical screen alone is not enough" in basis


def test_adverse_evidence_is_never_reported_as_pass():
    for screen in ("PASS", "CONDITIONAL PASS", "FAIL"):
        assert audit.overall_finding(screen, review(audit.REVIEW_EVIDENCE))[0] == "FAIL"
        assert audit.overall_finding("FAIL", review(audit.REVIEW_NO_EVIDENCE))[0] == "FAIL"
        assert audit.overall_finding("FAIL", review(audit.REVIEW_NOT_PERFORMED))[0] == "FAIL"


def test_without_a_manual_review_the_screen_alone_never_gives_a_pass():
    for screen in ("PASS", "CONDITIONAL PASS"):
        finding, basis = audit.overall_finding(screen, review(audit.REVIEW_NOT_PERFORMED))
        assert finding == "UNRESOLVED" and "cannot settle origin versus destination" in basis


def test_a_manual_review_must_use_a_known_conclusion():
    with pytest.raises(ValueError):
        audit.ManualReview("looks fine", "2026-10-22", ())
    assert audit.S08A_REVIEW.conclusion == audit.REVIEW_NO_EVIDENCE
    assert audit.S08A_REVIEW.reviewed_on == "2026-10-22" and len(audit.S08A_REVIEW.basis) >= 4


def test_marker_diameter_units_and_scale():
    d = {v: audit.marker_diameter_m(v) for v in (0.05, 0.20, 0.80)}
    assert d[0.20] == pytest.approx(2 * d[0.05]) and d[0.80] == pytest.approx(2 * d[0.20])
    assert audit.marker_diameter_m(0.6, audit.ORIGINAL_XG_AREA) > audit.marker_diameter_m(0.6)
    ratio = audit.marker_diameter_m(0.5, 2000.0) / audit.marker_diameter_m(0.5, 500.0)
    assert ratio == pytest.approx(2.0)  # area, not diameter, is proportional to the scale factor
    assert audit.marker_diameter_m(0.6, audit.ORIGINAL_XG_AREA) > 5.5, "wider than the six-yard box"


# --- reading the archive: development events only -------------------------------------


class GuardedEvent(dict):
    """An event of a held-out match: reading anything but ``matchId`` is an error."""

    def __init__(self, event, log):
        super().__init__(event)
        self.log = log

    def __getitem__(self, key):
        if key != "matchId":
            self.log.append(key)
            raise AssertionError(f"held-out event: field {key!r} was read")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key != "matchId":
            self.log.append(key)
            raise AssertionError(f"held-out event: field {key!r} was read")
        return super().get(key, default)


class FakeSource:
    """Stands in for ``WyscoutSource``: all events appear in the England file, none elsewhere."""

    def __init__(self, events, held_out_matches, log):
        self.events, self.held_out, self.log = events, held_out_matches, log

    def iter_events(self, member):
        if member != "events_England.json":
            return
        for event in self.events:
            if event["matchId"] in self.held_out:
                yield GuardedEvent(event, self.log)
            else:
                yield dict(event)


def make_events(rows, swap_first=None):
    events = []
    for r in rows:
        positions = [{"x": r.x_pct, "y": r.y_pct}, {"x": 100, "y": 100}]
        if r.event_id == swap_first:
            positions.reverse()
        tags = [{"id": BODY_TAG[r.body_part_class]}] + ([{"id": 101}] if r.is_goal else [])
        events.append(
            {
                "id": r.event_id + 1_000_000_000_000, "matchId": r.match_id, "teamId": r.team_id,
                "playerId": r.player_id, "eventId": 8, "subEventId": 85, "eventName": "Pass",
                "subEventName": "Simple pass", "matchPeriod": r.period,
                "eventSec": r.event_sec - 3.0, "tags": [{"id": 1801}],
                "positions": [{"x": 50, "y": 50}, {"x": r.x_pct, "y": r.y_pct}],
            }
        )  # fmt: skip
        events.append(
            {
                "id": r.event_id, "matchId": r.match_id, "teamId": r.team_id,
                "playerId": r.player_id, "eventId": 10, "subEventId": 100, "eventName": "Shot",
                "subEventName": "Shot", "matchPeriod": r.period, "eventSec": r.event_sec,
                "positions": positions, "tags": tags,
            }
        )  # fmt: skip
    return events


def test_only_development_events_are_examined_and_links_use_the_preceding_pass():
    rows = s05_rows(n_dates=6, matches_per_date=1, shots_per_match=4, quarantined_per_match=0)
    events = make_events(rows)
    held_out = {rows[-1].match_id}
    log: list[str] = []
    source = FakeSource(events, held_out, log)
    development = {r.match_id for r in rows} - held_out
    shots, context, links = audit.read_shots_and_context(source, development, {rows[0].match_id})
    assert log == []
    assert set(shots) == {r.event_id for r in rows if r.match_id in development}
    assert all(e["matchId"] == rows[0].match_id for e in context[rows[0].match_id])
    assert len(context[rows[0].match_id]) == 8  # 4 shots and 4 passes
    assert set(links) == set(shots) and all(v == 0.0 for v in links.values())
    guarded = next(e for e in source.iter_events("events_England.json") if e["matchId"] in held_out)
    assert isinstance(guarded, GuardedEvent)
    with pytest.raises(AssertionError, match="held-out event"):
        guarded.get("positions")  # the trap itself works
    assert log == ["positions"]


# --- the audit end to end -------------------------------------------------------------


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    folder = tmp_path_factory.mktemp("s08a")
    rows = s05_rows()
    csv_path, manifest_path, rows = write_dataset(folder / "data", rows)
    nothing = folder / "absent.md"
    ev = evaluate_nonlinear(csv_path, manifest_path, folder / "split", 10, nothing, nothing)
    reference = {
        m: {"log_loss": s.log_loss, "brier": s.brier, "auc": s.auc, "n": s.n, "goals": s.goals}
        for m, s in ev.scores["validation"].items()
    }
    for rel, content in {
        "reports/final_test_evaluation.md": "report",
        "results/final_test_summary.json": "{}",
        "reports/final_test_execution_log.jsonl": "{}",
    }.items():
        path = folder / "s07" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    val_dates = sorted(d for d, s in ev.plan.date_split.items() if s == "validation")
    case = min(r.match_id for r in rows if r.match_datetime_utc[:10] == val_dates[0])
    test_dates = held_out_dates(csv_path, manifest_path)
    held_out_matches = {r.match_id for r in rows if r.match_datetime_utc[:10] in test_dates}
    return SimpleNamespace(
        folder=folder, csv=csv_path, manifest=manifest_path, rows=rows, ev=ev, case=case,
        reference=reference, nothing=nothing, test_dates=test_dates,
        held_out_matches=held_out_matches,
    )  # fmt: skip


def run(world, rows_reader=iter_csv_rows, events=None, log=None, **kwargs):
    events = events if events is not None else make_events(world.rows)
    source = FakeSource(events, world.held_out_matches, log if log is not None else [])
    return audit.run_audit(
        csv_path=world.csv, manifest_path=world.manifest, split_dir=world.folder / "split",
        s03_report=world.nothing, protocol=world.nothing, source=source, rows=rows_reader,
        case_match_id=world.case, reference_scores=world.reference,
        s07_root=world.folder / "s07", **kwargs,
    )  # fmt: skip


@pytest.fixture(scope="module")
def result(world):
    mp = pytest.MonkeyPatch()
    mp.setattr(s04_helpers, "SENSITIVE", s04_helpers.SENSITIVE | {"team_id", "side", "player_id"})
    csv_trap: list[str] = []
    event_trap: list[str] = []

    def reader(path):
        for row in iter_csv_rows(path):
            yield GuardedRow(row, row["match_datetime_utc"][:10] in world.test_dates, csv_trap)

    out = run(world, reader, log=event_trap)
    mp.undo()
    return SimpleNamespace(r=out, csv_trap=csv_trap, event_trap=event_trap)


def test_the_audit_touches_no_held_out_event_field_or_shot_table_column(result):
    assert result.event_trap == [] and result.csv_trap == []


def test_every_development_shot_reconciles_with_the_source_positions(result, world):
    r = result.r
    dev_eligible = [
        r_ for r_ in world.rows if r_.eligible and r_.match_id not in world.held_out_matches
    ]
    assert r.n_development == len(dev_eligible) == r.reconciled
    assert r.mismatches == [] and r.unmatched == 0
    assert r.goal_flag_mismatch == 0 and r.body_part_mismatch == 0


def test_a_swapped_positions_array_is_detected_and_fails_the_finding(world):
    victim = next(
        r for r in world.rows
        if r.eligible and r.match_id not in world.held_out_matches and r.x_pct != 100
    )  # fmt: skip
    out = run(world, events=make_events(world.rows, swap_first=victim.event_id))
    assert [m["event_id"] for m in out.mismatches] == [victim.event_id]
    assert out.screen == "FAIL" and "differ from positions[0]" in out.screen_reasons[0]
    assert out.finding == "FAIL", "a positive manual review cannot override a failed screen"


def test_tables_aggregate_to_the_development_totals_only(result, world):
    r = result.r
    total_goals = sum(
        x.is_goal for x in world.rows if x.eligible and x.match_id not in world.held_out_matches
    )
    for rows in (r.bands_overall, r.zones, r.x_decile_table):
        assert sum(x[1] for x in rows) == r.n_development
        assert sum(x[2] for x in rows) == total_goals
    assert sum(sum(x[1] for x in rows) for rows in r.bands_by_league.values()) == r.n_development
    assert sum(sum(x[1] for x in rows) for rows in r.bands_by_body.values()) == r.n_development
    assert sum(x[1] for x in r.second_position) == len(
        [x for x in world.rows if x.match_id not in world.held_out_matches]
    )  # every development regular shot, quarantined ones included


def test_the_case_match_table_matches_the_shot_table_row_for_row(result, world):
    mine = [x for x in world.rows if x.match_id == world.case and x.eligible]
    assert len(result.r.case) == len(mine)
    keys = [c["sort"] for c in result.r.case]
    assert keys == sorted(keys), "events are sorted by period and time"
    by_event = {x.event_id: x for x in mine}
    for c in result.r.case:
        row = by_event[c["event"]["id"]]
        assert c["origin"] == (row.x_pct, row.y_pct) == c["positions"][0]
        assert c["goal"] == bool(row.is_goal) and c["team"] == row.team_id
        assert c["distance"] == pytest.approx(row.distance_m, abs=1e-5)
        assert math.radians(c["angle_deg"]) == pytest.approx(row.angle_rad, abs=1e-6)
        assert 0.0 < c["xg"] < 1.0
        assert c["flags"] == audit.diagnostic_flags(row.x_pct, row.y_pct)


def test_goal_line_records_are_listed_with_their_neighbours_and_only_from_development(
    result, world
):
    listed = {g["event"]["id"] for g in result.r.goal_line}
    expected = {
        x.event_id for x in world.rows
        if x.eligible and x.x_pct == 100 and x.match_id not in world.held_out_matches
    }  # fmt: skip
    assert listed == expected
    assert all(g["split"] in ("train", "validation") for g in result.r.goal_line)
    assert all(len(g["context"]) >= 1 for g in result.r.goal_line)


def test_the_report_reconciles_with_the_result_and_names_no_held_out_identifier(result, world):
    text = audit.render_report(result.r)
    assert f"## Finding: {result.r.finding} (" in text
    assert "### Mechanical screen (computed, reproducible)" in text
    assert "### Manual review (a dated judgement, not computed)" in text
    assert result.r.finding_basis in text
    assert "two passes" in text and "one pass" not in text
    assert f"{result.r.n_development:,} eligible regular shots" in text
    held_out_events = {
        x.event_id for x in world.rows if x.match_datetime_utc[:10] in world.test_dates
    }
    for event_id in held_out_events:
        assert str(event_id) not in text
    for match_id in world.held_out_matches:
        assert str(match_id) not in text
    for needle in ("positions[0]", "CC BY 4.0", "10.6084/m9.figshare.c.4415000.v5",
                   "External corroboration note", "Limits of this audit"):  # fmt: skip
        assert needle in text


def test_s07_artefacts_are_hashed_before_and_after_and_a_change_aborts(world, monkeypatch):
    out = run(world)
    assert out.s07_before == out.s07_after and set(out.s07_before) == {"report", "summary", "log"}
    calls = {"n": 0}
    real = audit.s07_hashes

    def changing(root=Path(".")):
        calls["n"] += 1
        hashes = real(root)
        return hashes if calls["n"] == 1 else {**hashes, "log": "0" * 64}

    monkeypatch.setattr(audit, "s07_hashes", changing)
    with pytest.raises(audit.AuditError, match="S07 artefact changed"):
        run(world)


def test_a_case_match_outside_the_development_partition_is_refused(world):
    held_out = next(iter(world.held_out_matches))
    with pytest.raises(audit.AuditError, match="not a development match"):
        audit.run_audit(
            csv_path=world.csv, manifest_path=world.manifest, split_dir=world.folder / "split",
            s03_report=world.nothing, protocol=world.nothing,
            source=FakeSource(make_events(world.rows), world.held_out_matches, []),
            case_match_id=held_out, reference_scores=world.reference,
            s07_root=world.folder / "s07",
        )  # fmt: skip


def test_the_audit_does_not_call_the_final_evaluation(world, monkeypatch):
    from expected_goals_lab import final_evaluation as fe

    def explode(*args, **kwargs):
        raise AssertionError("final-test code was called")

    for name in ("execute_final_evaluation", "load_final_test_partition", "main"):
        monkeypatch.setattr(fe, name, explode)
    assert fe.REAL_EXECUTION_ENABLED is False
    run(world)


# --- the committed report, read as text -----------------------------------------------

REPORT = REPO / "reports/shot_origin_audit.md"


def table_rows(text, heading):
    """Body rows of the first Markdown table after ``heading``."""
    lines = text.split(heading, 1)[1].splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("|"))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


@pytest.mark.skipif(not REPORT.exists(), reason="audit report not generated")
def test_committed_report_tables_are_internally_consistent():
    text = REPORT.read_text(encoding="utf-8")
    n = int(
        re.search(r"Development rows audited: ([\d,]+) eligible", text).group(1).replace(",", "")
    )
    overall = table_rows(text, "### 3.1 Origin x by band, all leagues")
    assert sum(int(r[1]) for r in overall) == n
    assert all(int(r[1]) == int(r[2]) + int(r[3]) for r in overall)
    leagues = re.findall(r"\*\*(\w+ first division)\*\*\n\n((?:\|.*\n)+)", text)
    assert len(leagues) == 5
    assert sum(int(line.split("|")[2]) for _, tbl in leagues for line in tbl.splitlines()[2:]) == n
    zones = table_rows(text, "### 3.3 Goal-line and centre-of-goal positions")
    assert sum(int(r[1]) for r in zones) == n
    case = table_rows(text, "## 2. Match 2516947")
    assert len(case) == 20
    for row in case:
        positions, selected = row[7], row[8]
        first = positions.split(" ; ")[0]
        assert first == selected, row
    assert "## Finding:" in text and "### Mechanical screen" in text
    assert "### Manual review (a dated judgement, not computed)" in text


@pytest.mark.skipif(not REPORT.exists(), reason="audit report not generated")
def test_committed_report_records_unchanged_s07_hashes_and_a_closed_gate():
    text = REPORT.read_text(encoding="utf-8")
    rows = table_rows(text, "## 5. Frozen artefacts")
    assert len(rows) == 3 and all(r[1] == r[2] for r in rows)
    for name, path in (("report", "reports/final_test_evaluation.md"),
                       ("summary", "results/final_test_summary.json"),
                       ("log", "reports/final_test_execution_log.jsonl")):  # fmt: skip
        file = REPO / path
        if file.exists():
            expected = pdata.sha256_file(file)
            assert next(r for r in rows if r[0] == name)[1] == f"`{expected}`"
    assert dataclasses.is_dataclass(pf.MatchShots)


def clean_screen(monkeypatch):
    """The synthetic table has many goal-line shots; force a clean screen to test the review."""
    monkeypatch.setattr(audit, "mechanical_screen", lambda *a: ("PASS", ["all PASS criteria met"]))


def test_a_clean_screen_and_a_positive_review_give_a_labelled_judgement(world, monkeypatch):
    clean_screen(monkeypatch)
    out = run(world)
    assert out.finding == "PASS" and "review judgement" in out.finding_basis
    text = audit.render_report(out)
    assert "## Finding: PASS (evidence-based review judgement" in text
    assert "**PASS.** The screen checks reconciliation" in text


def test_an_audit_run_without_a_manual_review_reports_unresolved_not_pass(world, monkeypatch):
    clean_screen(monkeypatch)
    out = run(world, review=audit.ManualReview(audit.REVIEW_NOT_PERFORMED, "2026-10-22", ()))
    assert out.finding == "UNRESOLVED"
    text = audit.render_report(out)
    assert "## Finding: UNRESOLVED (" in text and "## Finding: PASS" not in text
    assert "Manual review (a dated judgement, not computed): not performed" in text


def test_an_audit_run_with_adverse_review_fails_even_when_the_screen_is_clean(world, monkeypatch):
    clean_screen(monkeypatch)
    out = run(world, review=audit.ManualReview(audit.REVIEW_EVIDENCE, "2026-10-22", ("x",)))
    assert out.finding == "FAIL"
    assert "## Finding: FAIL (the manual review found evidence" in audit.render_report(out)
