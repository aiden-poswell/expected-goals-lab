# Wyscout data audit (S01B)

Status: complete, awaiting review. No model has been fitted and no S02 shot table exists.

Provenance of the numbers below:

- **[CLI]** reproduced by `uv run python -m expected_goals_lab.wyscout_audit` and printed in `reports/wyscout_data_audit.md` (run of 2026-10-20; a second cached run was byte-identical apart from the timestamp).
- **[explored]** observed with throwaway scripts during the audit and *not* reproduced by the CLI. They are real observations but a reviewer cannot regenerate them from the repository.
- **[not verified]** things I could not or did not check.

## 1. Source, version, licence

**Collection:** Pappalardo, L. & Massucco, E. (2019). *Soccer match event dataset.* figshare. Collection, <https://doi.org/10.6084/m9.figshare.c.4415000.v5> (concept DOI <https://doi.org/10.6084/m9.figshare.c.4415000>). Figshare API, read 2026-10-20: version 5 is the latest of five collection versions; published 2020-01-28.

**Pinned files** (all read from the Figshare API on 2026-10-20; every file is size- and MD5-verified on each load by `wyscout.py`):

| File | Article (version) | Figshare file ID | Bytes | MD5 |
|---|---|---|---|---|
| `events.zip` | 7770599 (v1, only version) | 14464685 | 77,323,413 | `7c20e8647e7eda58d7838a0c7b1ec6ab` |
| `matches.zip` | 7770422 (v1, only version) | 14464622 | 645,097 | `51d80beb17480919f69a53a0152c2d71` |
| `competitions.json` | 7765316 (v4, latest of 4) | 15073685 | 1,209 | `3dc210a4805dda5337b0ff9f7eaa407a` |
| `eventid2name.csv` | 11743836 (v1) | 21385245 | 1,001 | `46daf16100ece0c743eedc9adcfea162` |
| `tags2name.csv` | 11743818 (v1) | 21385239 | 1,754 | `e7acb14918d00e40c80a898b1da8fc39` |

Teams, players, coaches and referees files were not downloaded; they are not needed for this slice. Download URL pattern: `https://ndownloader.figshare.com/files/<file ID>`. The first download of all five files took about 30 s; the CLI never re-downloads a verified cached copy.

**Licence evidence**

- Every article above is marked **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>) by the Figshare API. I only read the five articles listed; the other seven in the collection (for example `PlayeRanks`, replication code, the data paper) were not examined.
- The primary paper states "The data sets are released under the CC BY 4.0 License and are publicly available on figshare." Pappalardo et al. (2019), *A public data set of spatio-temporal match events in soccer competitions*, Scientific Data 6:236, <https://doi.org/10.1038/s41597-019-0247-7> (open access, PMC6817871; text read via Europe PMC on 2026-10-20). The paper is itself CC BY 4.0, and its licence statement requires "appropriate credit to the original author(s) and the source", "a link to the Creative Commons license", and an indication "if changes were made".
- The Figshare collection description asks users to cite the Scientific Data paper and Pappalardo et al. (2019), *PlayeRank*, ACM TIST 10(5), Article 59, <https://doi.org/10.1145/3343172>.
- I did **not** use any GitHub mirror. This audit reads only from Figshare, so nothing here says anything about a mirror's licence.

**Attribution to carry in any publication** (derived from the above; not legal advice): cite the collection DOI and the Scientific Data paper, name the authors (Pappalardo, Massucco and co-authors) as the source, link to the CC BY 4.0 licence, state that the data were collected by Wyscout, and state what was changed (filtering, coordinate conversion, derived features).

**What I could not verify.**

- The paper says the data were "collected and provided by Wyscout", and one author is affiliated with Wyscout. I found no statement in the paper or on Figshare of Wyscout's own redistribution terms, and I did not review Wyscout's commercial terms. CC BY 4.0 is granted by the dataset's authors; whether the rights chain to Wyscout supports it is **[not verified]**. This is a residual risk, not a finding of a problem.
- Nothing here is legal advice, and I have not approved publication. The decision belongs to the owner.

## 2. Published corpus (from metadata) [CLI]

1,941 matches in 7 competitions, all `status` = `Played`, all one season each. Match counts equal Table 1 of the paper.

| Competition (wyId) | Type | Matches | Teams | Dates (UTC) | Durations |
|---|---|---|---|---|---|
| English first division (364) | club | 380 | 20 | 2017-08-11 to 2018-05-13 | Regular 380 |
| French first division (412) | club | 380 | 20 | 2017-08-04 to 2018-05-19 | Regular 380 |
| German first division (426) | club | 306 | 18 | 2017-08-18 to 2018-05-12 | Regular 306 |
| Italian first division (524) | club | 380 | 20 | 2017-08-19 to 2018-05-20 | Regular 380 |
| Spanish first division (795) | club | 380 | 20 | 2017-08-18 to 2018-05-20 | Regular 380 |
| European Championship 2016 (102) | international | 51 | 24 | 2016-06-10 to 2016-07-10 | Regular 46, ExtraTime 2, Penalties 3 |
| World Cup 2018 (28) | international | 64 | 32 | 2018-06-14 to 2018-07-15 | Regular 59, ExtraTime 1, Penalties 4 |

Matches per team are 38, 38, 34, 38 and 38 (England, France, Germany, Italy, Spain), consistent with double round robins; I did not check each pairing. The five leagues are 1,826 matches. The competition files are named for countries (`events_England.json`) and the leagues are the top divisions; I did not verify the 2017/18 season label beyond the dates above.

## 3. Archive-wide count scan [CLI]

Every event file was read once. This keeps *counts only*; it is not a shot extraction and builds no shot table.

- Event counts equal the paper's Table 1 in all seven files (643,150 England; 632,807 France; 519,407 Germany; 647,372 Italy; 628,659 Spain; 78,140 Euro 2016; 101,759 World Cup).
- Unknown `matchId`: 0. Event `teamId` not among the match's two teams: 0. Duplicate event IDs within a file: 0 (cross-file duplicates not checked). Matches in metadata without events: 0. `eventSec` going backwards within a (match, period): 0.
- Every event in every file has exactly the same 12 keys: `eventId, eventName, eventSec, id, matchId, matchPeriod, playerId, positions, subEventId, subEventName, tags, teamId`. There are no missing keys to impute; missingness lives in values and tags (section 7).

**Shot-type events** (goal-tagged in brackets; shootout = `matchPeriod` `P`):

| League | Open-play shots | Free-kick shots | Penalties in play |
|---|---|---|---|
| England | 8,451 (914) | 350 (18) | 80 (56) |
| France | 8,327 (873) | 521 (26) | 129 (99) |
| Germany | 6,898 (747) | 299 (18) | 93 (68) |
| Italy | 8,806 (853) | 415 (36) | 126 (89) |
| Spain | 7,979 (884) | 453 (28) | 113 (81) |
| **Five leagues** | **40,461 (4,271)** | **2,038 (126)** | **541 (393)** |

Derived from those tallies: 42,499 non-penalty in-play attempts in the five leagues, 4,397 of them goal-tagged (10.3%), against 541 penalties of which 393 are goal-tagged (72.6%). These are tallies of tagged events, not validated shot records. The tournaments add 2,617 open-play shots, 171 free-kick shots, 41 penalties in play and 76 shootout kicks (54 goal-tagged).

## 4. Sample [CLI]

37 matches: 6 per domestic league, evenly spaced through the season in `(dateutc, wyId)` order using index `(2i+1)*n // (2k)` (deterministic, input-order independent), plus all 7 international matches decided by a penalty shootout (World Cup 4, Euro 3). They contain 65,503 events and 1,011 shot-type events. Match IDs are listed in the report. The sample exists to validate event-level behaviour; it is far too small for any statement about rates.

## 5. Event taxonomy

From `eventid2name.csv`, `tags2name.csv`, the Wyscout API documentation (<https://support.wyscout.com/matches-wyid-events>) and the data:

| Concept | How it appears |
|---|---|
| Open-play shot | `eventId` 10 / `subEventId` 100 "Shot" |
| Free-kick shot | `eventId` 3 / `subEventId` 33 "Free kick shot" |
| Penalty | `eventId` 3 / `subEventId` 35 "Penalty" (a *Free Kick* event, not a Shot) |
| Shootout kick | A penalty with `matchPeriod` `P`. Present only in the tournament files: 37 (Euro) + 39 (World Cup) |
| Headers | Not separable. Tag 403 "head/body" covers headers and other body parts. Tags 401/402 are left/right foot |
| Goal | Tag 101 |
| Own goal | Tag 102 on an ordinary event of the player who scored it, logged for the scorer's own team |
| Direct corner goal | Tag 101 on a Corner event (3/30), with no shot event: 3 cases |

Traps found in real data [CLI]:

1. **Goal tag 101 also sits on goalkeeper "Save attempt" events** (4,643 + 638 of them), logged for the *conceding* team. Counting every 101 double-counts goals and credits them to the wrong side. Tallies by event type: Save attempt 5,281; Shot 4,492; Penalty 477; Free kick shot 137; Corner 3.
2. **Own-goal tag 102** appears on Touch (128), Clearance (26), Head pass (11) and Simple pass (3) events. They are not shots.
3. Rebuilding each team's goals as "101 on any event except Save attempt, outside the shootout, plus opponent's 102" and comparing with the match record (`score` for regular-duration matches, `scoreET` for extra-time and shootout matches) agrees for **3,881 of 3,882 team-matches** (3,651 of 3,652 domestic; all 230 tournament). Absence of tag 101 can therefore be read as "not a goal" for the purpose of labelling, on the evidence of this reconciliation. I did not extend that inference to any other tag.
4. The one disagreement: match 2499781 (Chelsea – Manchester City, 0–1): the events give Manchester City no goal. [explored] The Chelsea goalkeeper's Save attempt carries tag 101, but no attacking event does, so the scoring shot is not tagged as a goal. This is a concrete example of label noise; there may be others that cancel out in the reconciliation.
5. In domestic matches `scoreET` is 0 even when goals were scored; use `score` there [explored; see `published_score`].

Eligible non-penalty scoring attempt (proposal; S02 finalises): a shot-type event of kind open play or free kick, `matchPeriod` in `1H/2H/E1/E2` (not `P`), with a valid integer start position, whose match is not flagged for coordinate anomalies. Penalties and shootout kicks are excluded. Whether blocked shots and free-kick shots belong in the main cohort is an S02 decision; blocked shots are shot events (24.5% of sampled attempts carry tag 2101) [CLI].

## 6. Coordinates [CLI unless noted]

- **Documentation.** The Wyscout API page says `x`, `y` are integer "percentage from the left corner of the attacking team". The paper adds that x measures nearness to the opponent's goal and y nearness to the right side of the field, "from the perspective of the attacking team". Both are brief.
- **Data checks.**
  - Penalties in play (582): mean start position (89.3, 50.0), most common (89, 50) and (90, 50). So: scale is 0–100, the attacked goal is at x = 100, and the goal's centre line is y = 50. No shot start position in the whole archive is outside 0–100 or non-integer.
  - Open-play shots by x band: goal-tag share rises from 2.3% (70–80) to 8.5% (80–90) to 19.3% (90–100). That is consistent with x = 100 being the attacked goal and is not a model result.
- **No flipping is needed to put both teams on one convention**: each event is already expressed from the acting team's perspective, with x = 100 as the goal being attacked (the data support this, above). I did not apply any flip.
- **y direction is unresolved.** The documentation's "left corner of the attacking team" does not say whether y = 0 is that team's left or right touchline, and I did not establish it. Distance and goal-mouth angle are symmetric about y = 50 so this does not matter for them; it would matter for any left/right feature such as foot versus side.
- **Placeholder second position.** Shot-type events carry two entries in `positions`; the second is (0,0) for 25,100 shots, (100,100) for 20,260 and another value for 585. The API documentation says positions record only the start "if not a movement". Use only the first entry. [explored] The two placeholder values both occur with goals and non-goals, so I found no use for it and treat it as not an end location.
- **Resolution.** Integers on a 0–100 scale: with a 105 × 68 m pitch one unit is about 1.05 m along the pitch and 0.68 m across it. Penalty mean x = 89.3 with the 11 m mark gives an implied length of about 103 m, i.e. consistent with a 100–110 m pitch but not a measurement of it.
- **Converting to metres (not implemented).** With pitch length L and width W: `x_m = x/100·L`, `y_m = y/100·W`, goal centre at `(L, W/2)`, posts at `y_m = W/2 ± 3.66`. Distance and the angle subtended by the posts follow directly. Assumptions to state: L = 105, W = 68 (the usual reference size) because the dataset gives no per-stadium dimensions; the match record names a venue only; the scale is linear; 3.66 m is half the goal width. S02 should report sensitivity to L ∈ {100, 105, 110} and W ∈ {64, 68, 75}, and should expect integer quantisation to matter most for close-range angle.
- **Orientation anomalies.** 100 open-play shots have x < 50 and 13 have x < 10 (3 goal-tagged). The 13 sit in 6 matches (2500923, 2516742, 2516958, 2516989, 2565753, 2565775); 12 are second-half. [explored] In Hannover 96 – RB Leipzig (2516989), six shots by one team (teamId 2975) are at x = 3–8, two of them goal-tagged; in Sevilla – Girona (2565775) a goal-tagged shot is recorded at x = 1, y = 51. I read this as the attacking direction being recorded the wrong way round for part of a match. That is an inference. A flipped shot is detectable only when its true position is close to the opponent's goal (a true x of 95 appears as 5); flipped shots from further out look like ordinary long shots, so there may be undetected ones. [not verified]

## 7. Missingness and data quality [CLI unless noted]

- **Sample (1,011 shot-type events, 37 matches):** 0 failing identity validation (`matchId`, `teamId`, `id`); 37 of 37 matches have event team IDs equal to the two teams in the match record; 0 shooters absent from the team's lineup or bench; 0 duplicate shot IDs; 0 malformed player, period, time, position, tag or body-part values.
- **Body part:** every sampled non-penalty attempt carries exactly one of 401/402/403 (right foot 50.5%, left 32.3%, head/body 17.2%). Archive-wide [explored], every shot-type event has exactly one of the three. Absence was *not* assumed to mean "not left foot"; the parser returns `None` for none or conflicting tags.
- **Accuracy tags:** 1801/1802 are exclusive and present on all sampled attempts (31.8% accurate).
- **`playerId` 0:** 3 shot events archive-wide (England 2500037, Italy 2576175, Spain 2565756), parsed as unknown.
- **Offside events** carry `subEventId` `''` (8,182) instead of an integer. [explored]
- **Match lineups:** `goals` is a string and can be the text `null` for bench players; `ownGoals` is a string with values '0', '1' and '2' where '2' occurs on 7,053 players, which cannot be own-goal counts (only 168 own-goal-tagged events exist). Do not use the lineup `ownGoals` or `goals`. [explored]
- Some `label` strings contain escaped unicode literally (for example `Málaga`). [CLI output]
- Tag 201 "opportunity" is on about 71% of open-play shots (30,458 of 43,078, all files) [explored]. Its definition is not given, and with that prevalence it is not a dependable "big chance" flag.

## 8. What is available at shot time

Per the schema above, a shot event provides: start position; kind (open play, free kick, penalty); body part (left foot, right foot, head/body); period and second; shooter, team, opponent and match. Because events are time-ordered within each match period, earlier events in the same match can supply context strictly *before* the shot: the preceding pass subtype (cross, head pass, high pass, launch, smart pass), the "through" (901) and "key pass" (302) tags where present on that pass, its start and end positions, elapsed time since it, running score from earlier goal events, team/opponent identity. Ordering is verified within (match, period); whether the file is contiguous by match is **[not verified]**, so S02 should sort explicitly.

**Not available** in this dataset: freeze frames (defender and goalkeeper positions), shot end location, pressure, first-time flag, one-on-one and open-goal flags, shot technique (volley, lob and so on), separate head/other-body-part, possession and play-pattern IDs, and the linked key pass ID.

**Must not be used as features** (outcome or post-shot information): the shot's own Goal tag 101; accuracy tags 1801/1802; Blocked tag 2101; goal-mouth placement tags 1201–1223; the second position entry; any Save attempt event linked to the shot; later events. Tag 1901 "counter attack" and 201 "opportunity" are tagger judgements whose timing relative to the outcome is unknown; exclude them until their definition is established.

**External benchmark.** None found. No event carries an xG field (all events share the same 12 keys), and the tag list has no xG tag. I opened only the five articles listed in section 1; I did not inspect the `PlayeRanks` article. Treat "no provider xG" as true of the files audited. A benchmark derived from any other provider would bring that provider's terms into a published comparison and is not recommended without clarifying them.

## 9. Recommended modelling and evaluation approach (not executed)

1. **Cohort.** Non-penalty in-play attempts from the five domestic leagues: about 42.5k tagged events and 4.4k goals (section 3, before validation). Open-play and free-kick attempts get a `kind` feature or are modelled separately. Penalties (541, 72.6% goal-tagged) are a separate trivial problem or excluded. Shootout kicks are excluded.
2. **Chronological split by match date.** Order all domestic matches by `dateutc` pooled across the five leagues (they share a calendar); take roughly the first 60% for training, the next 20% for validation (model and hyperparameter selection), and the final 20% as a test partition touched once. No match appears in two partitions. With about 20% of attempts the test set holds on the order of 8.5k attempts and 880 goals, enough for stable log loss and calibration intervals; the exact figures come from S02.
3. **Competition-held-out evaluation.** Leave one league out (train and validate on four, test on the fifth; 7.2k to 9.2k attempts and 765 to 932 goals each) is supported by shot counts. It measures *distribution shift between leagues* within one season, not prediction of the future; keep it separate from the chronological result and do not report a single mixed number.
4. **Season limit.** All five leagues are one season (2017/18). A chronological holdout within it is possible; **out-of-season generalisation cannot be established** with this data. The tournaments (2016, 2018) are different populations and years, small (about 2.6k open-play shots), and are best treated as an optional exploratory shift check, not as proof of temporal generalisation.
5. **Baselines.** (a) Unconditional training goal rate; (b) logistic on distance; (c) logistic on distance and angle; (d) plus kind and body part. Add context features only if they beat (c)/(d) on validation by more than the uncertainty, and say what was tried.
6. **Metrics.** Log loss (primary), Brier score, calibration (reliability bins plus calibration intercept and slope), AUC as secondary only. Uncertainty by match-clustered bootstrap. Never accuracy.
7. **Data hygiene before fitting.** Handle the orientation anomalies (exclude or sensitivity-test flagged matches), the untagged-goal match, `playerId` 0, and coordinate-to-metre sensitivity (section 6). Count and report exclusions.
8. **Caveats to carry.** The same teams and players appear in every partition, and the labels come from human tagging. A tagging bias, for instance in what counts as a shot, cannot be separated from true behaviour.

## 10. Limitations of this audit

- Shot-level validation covers 37 matches. Archive-wide statements are tallies of tags and IDs.
- The licence chain beyond the dataset authors is unverified (section 1).
- The orientation-flip explanation is an inference, and its extent is unknown.
- Single season; no external xG benchmark.
- Exploratory observations (marked) are not reproducible from the repository.

## 11. Proposed next slice (not started)

If the owner accepts `docs/DATA_SOURCE_DECISION.md`: S02-Wyscout, a canonical shot table for the five domestic leagues (IDs, match date, kind, tags needed for labels, start position, strictly pre-shot context), orientation-anomaly handling, coordinate conversion with the sensitivity analysis in section 6, and tests for symmetry, direction and edge cases. It needs a decision first on the open licence and scope questions in the decision document.
