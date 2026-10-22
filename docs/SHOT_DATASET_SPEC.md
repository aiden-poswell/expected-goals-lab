# Shot dataset specification (S02)

Status: implemented, awaiting review. Nothing has been split, fitted or evaluated.

Build: `uv run python -m expected_goals_lab.build_shots` (`--offline` uses only the verified cache). Output, all git-ignored: `data/processed/wyscout_regular_shots.csv` and `data/processed/wyscout_regular_shots.manifest.json`. The compact aggregate audit is committed as `reports/shot_dataset_audit.md`; its figures are the authority for counts and are regenerated on every build.

## 1. Source and attribution

- **Data:** Pappalardo, L. & Massucco, E. (2019). *Soccer match event dataset.* figshare. Collection. [doi:10.6084/m9.figshare.c.4415000.v5](https://doi.org/10.6084/m9.figshare.c.4415000.v5). Files are pinned by Figshare file ID, size and MD5 (`events.zip`, `matches.zip`, `competitions.json`) and verified on every load.
- **Paper:** Pappalardo, L., Cintia, P., Rossi, A., Massucco, E., Ferragina, P., Pedreschi, D. & Giannotti, F. (2019). A public data set of spatio-temporal match events in soccer competitions. *Scientific Data* 6:236. [doi:10.1038/s41597-019-0247-7](https://doi.org/10.1038/s41597-019-0247-7). The collection description also asks for the *PlayeRank* paper (ACM TIST 10(5), Article 59, [doi:10.1145/3343172](https://doi.org/10.1145/3343172)) to be cited.
- **Original data provider:** Wyscout collected and provided the underlying event data.
- **Licence of the upstream data:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as stated on the Figshare articles and in the paper. Reuse requires credit to the authors and source, a link to the licence, and an indication of changes.
- **Changes made here:** the data were filtered to five domestic leagues and to one event class, parsed into flat rows, quarantined by quality rules, and augmented with derived columns (metric coordinates, distance, angle). The derived dataset is a transformation prepared for this project.
- **Not claimed:** neither the dataset authors nor Wyscout endorse this project or its results. The CC BY grant is made by the dataset authors; the chain of rights from Wyscout to them was not independently verified (`docs/DATA_SOURCE_DECISION.md`).
- **This repository's own code** is a separate matter from the upstream data licence. No licence has been chosen for the code yet; that is an owner decision and the CC BY 4.0 terms above do not apply to it.
- Raw archives, caches and the shot table are never committed (`.gitignore`). The committed report holds aggregates, match IDs and event IDs only.

## 2. Cohort definition

| Class | Wyscout event | In the table | Counted in the report |
|---|---|---|---|
| Regular shot | `eventId` 10, `subEventId` 100 ("Shot") | one row each | yes |
| Free-kick shot | 3 / 33 | no | yes (events and goal-tagged) |
| Penalty | 3 / 35 | no | yes |
| Shootout kick | penalty with `matchPeriod` `P` | no (none in domestic files) | not applicable to domestic leagues |
| Own goal | tag 102 on an ordinary event (touch, clearance, pass) of the scorer's team | no, it is not a shot | yes, in the goal bridge |
| Goal from a corner kick | tag 101 on a 3/30 event | no | yes, as "other attacking events" |

Scope is the five domestic league files only (England, France, Germany, Italy, Spain, 2017/18; 1,826 matches). The World Cup and European Championship files are not read.

"Regular shot" is a provider label, not a claim of open play: a regular shot can follow a corner, throw-in or free kick. The table has no field for what preceded the shot, and none is invented.

## 3. The label

`is_goal` is 1 if tag 101 is present on the shot event itself and 0 if the tag list is valid and lacks it; it is empty (and the row is quarantined with `LABEL_UNKNOWN`) if the tag list is missing or malformed. Nothing else produces a label:

- Tag 101 on goalkeeper `Save attempt` events (logged for the conceding team) is never used.
- The final score and the lineup `goals` field are never used. They appear only in the match-level reconciliation below.
- Accuracy (1801/1802), blocked (2101), goal-mouth placement (1201-1223) and other post-shot tags are not written to the table at all.

**Match reconciliation.** For every match, goals rebuilt from tags (101 on non-keeper events outside the shootout, plus the opponent's 102 events) are compared with the published score. A match that disagrees has all its rows quarantined (`MATCH_SCORE_MISMATCH`). The scoreboard is never used to choose or change a label. At the audited archive version one match disagrees (see section 6).

## 4. Row status: eligible or quarantined

Every regular shot has a row. `eligible = 1` only if no exclusion reason applies. `exclusion_reasons` lists all that apply (semicolon-separated, in the precedence order below) and `primary_exclusion` is the first.

| Precedence | Reason | Rule |
|---|---|---|
| 1 | `MATCH_SCORE_MISMATCH` | the match's rebuilt goals differ from the published score |
| 2 | `INVALID_POSITION` | first position entry missing, malformed, non-integer or outside 0-100 |
| 3 | `SUSPECT_POSITION` | `x_pct < 20`, see below |
| 4 | `INVALID_PERIOD_OR_TIME` | `matchPeriod` not `1H`/`2H`, or `eventSec` missing or negative |
| 5 | `LABEL_UNKNOWN` | tag list missing or malformed |
| 6 | `GEOMETRY_ERROR` | distance or angle cannot be computed |

Flags that do **not** exclude: `UNKNOWN_PLAYER` (`playerId` 0, missing or malformed), `BODY_PART_UNKNOWN` (no or conflicting body-part tags), `OWN_HALF_POSITION` (`x_pct < 50`), `ON_GOAL_LINE` (`x_pct = 100`, where the angle is degenerate: 180 degrees between the posts, 0 outside them).

**Suspect positions.** A shot at `x_pct < 20` is implausibly far from the goal it was aimed at, usually followed by an opponent goalkeeper save or a goal. The report shows, for each such shot, the neighbouring same-team events, the team's other shots and a comparison with opponent events within 3 s under the recorded and the mirrored (`100-x, 100-y`) frame. The evidence is mixed, so **no coordinate is flipped or repaired**; the rows are quarantined with raw values intact. The cutoff of 20 is a judgment: it covers the 13 shots below 10 and the 5 in the same sequences between 10 and 20; the report tabulates the cohort and goal-rate sensitivity for cutoffs 10, 20, 30 and 50. Shots with a wrongly framed position at an ordinary-looking location cannot be detected this way.

## 5. Columns

| Column | Type | Source / definition | Role |
|---|---|---|---|
| `event_id` | int | event `id`; unique | identifier |
| `match_id` | int | event `matchId` (checked equal to the match record) | identifier |
| `competition_id`, `competition` | int, text | match record `competitionId`; `competitions.json` name | identifier |
| `season_id` | int | match record `seasonId` | identifier |
| `match_datetime_utc` | text | match record `dateutc`, `YYYY-MM-DD HH:MM:SS`; sorts chronologically | identifier |
| `team_id`, `opponent_team_id` | int | event `teamId`; the other team of the match | identifier |
| `side` | `home`/`away` | match record side of `team_id` | context, reserved |
| `player_id` | int or empty | event `playerId`; empty when 0 | identifier |
| `period`, `event_sec` | text, float | `matchPeriod`; `eventSec` (seconds into the period) | context, reserved |
| `x_pct`, `y_pct` | int | first entry of `positions`, raw, attack-relative, 0-100 | location source |
| `x_m`, `y_m` | float | `105 * x_pct / 100`, `68 * y_pct / 100` | location source |
| `distance_m` | float | straight line to the goal centre `(105, 34)` | **core feature** |
| `angle_rad`, `angle_deg` | float | angle subtended by the posts, in `[0, pi]` | `angle_rad` core feature |
| `body_part_class` | text | `left_foot`, `right_foot`, `head_or_body` (tags 401, 402, 403), else `unknown` | candidate feature |
| `is_goal` | 0/1/empty | section 3 | **target** |
| `eligible`, `exclusion_reasons`, `primary_exclusion`, `flags` | | section 4 | quality |

Not available and therefore not present: shot end location (Wyscout's second position is a placeholder for 98.7% of shot-type events and is not a recorded end point), freeze frames, technique, pressure, first-time and one-on-one flags, play pattern, provider xG. Tag 403 means `head/body`, not a confirmed header.

Encoding: UTF-8, LF line endings, fixed column order and float precision (`x_m`/`y_m`/`distance_m` 6 decimals, `angle_rad` 9, `angle_deg` 6, `event_sec` 3). Rows are ordered by match date and time, match ID, period, `event_sec` and event ID, so the file is byte-for-byte reproducible; the SHA-256 is recorded in the manifest and the report.

## 6. Counts

*Note (added in S04 review):* this section publishes whole-cohort eligible goal totals. Together with the train and validation totals reported in S03, they let a reader infer the aggregate goal count of the later chronological test window. That window is therefore not outcome-blind at the aggregate level, although no row-level test label is accessed.

Authoritative figures are in `reports/shot_dataset_audit.md` (run of 2026-10-21). In short: 3,071,395 events in 1,826 matches; 40,461 regular-shot events (4,271 goal-tagged), 2,038 free-kick shots (126) and 541 penalties (393) are counted. All 40,461 regular shots have rows. 36 are quarantined (18 for the score mismatch, 18 for suspect positions, no overlap), leaving **40,425 eligible rows with 4,268 goals (10.56%, 95% Wilson interval 10.26 to 10.86)**. The published goals of the five leagues (4,947) are bridged to tags as 4,271 regular-shot, 126 free-kick, 393 penalty, 3 corner and 153 own goals, plus 1 unexplained goal.

**Match 2499781 (Chelsea – Manchester City, 0–1).** The only Goal tag in the match is on the Chelsea goalkeeper's Save attempt (2H, 1281.0 s). Manchester City has no Goal-tagged attacking event. The lineup record credits player 38021 with the goal; that player's only shot-type event is a first-half free kick without a Goal tag, and the last event by that player before the goalkeeper's event is a "Simple pass" tagged not accurate that ends at the placeholder position (100, 100). This is consistent with the goal having been logged as a pass, but the data do not prove it, and the scoreboard cannot say which provider event, if any, was the shot. All 18 regular-shot rows of the match (none goal-tagged) are quarantined; no label was changed.

## 7. Feature allowlist and leakage rules (for S03 onward)

Encoded in `shot_table.COLUMN_ROLES` and tested; every column has exactly one role.

- **Allowed model inputs, S03 baseline:** `distance_m`, `angle_rad`.
- **Candidate, S04 only if justified and reported:** `body_part_class`. Missing body part stays `unknown`; it is never imputed from the outcome or from other rows.
- **Reserved, not allowed until a slice justifies them:** `side`, `period`, `event_sec`.
- **Never inputs:** `is_goal` and any provider tag, goal-mouth or accuracy information, blocked flag, keeper event, final score, lineup goal counts, second-position entry, anything from events after the shot, and provider xG (none exists in this dataset).
- **Identifiers** (`event_id`, `match_id`, `competition*`, `season_id`, `match_datetime_utc`, `team_id`, `opponent_team_id`, `player_id`) are for grouping, ordering, splitting and auditing. They are not encoded as features automatically.
- **Quality columns** define the cohort (`eligible = 1`) and must not be used to re-admit quarantined rows silently.
- No target-based imputation, scaling fitted before a split, or selection on the test set. Splits are S03's job; chronology uses `match_datetime_utc`.

## 8. Known limits

- One season (2017/18) and five leagues: no claim about other seasons or leagues.
- The pitch size is assumed (`docs/GEOMETRY_VALIDATION.md`); positions are integer percentages (about 1 m along and 0.7 m across the pitch).
- Suspect-position and mismatch handling is conservative quarantine, not correction; undetected errors remain possible.
- Labels depend on provider tagging, which is human and unaudited here.
