# Progress Report: Understanding the Risk Labels

## Scope

This report focuses on what the dataset calls “risky” before judging model performance. A positive label means that two reported aircraft positions are within 5 nautical miles horizontally and 1,000 feet vertically in one frame. It is a proximity rule not a confirmed near-collision or proof that the aircraft were on a collision course.

The analysis uses a 58-minute Southern California backfill from September 25 and four roughly 30-minute samples from October 7 (6:39 AM, 10:18 AM, 2:30 PM, and 7:39 PM). The additional sessions help compare sampled times, but all four are from one day.

## How rare are positive cases?

The original snapshot had 55 positives among 8,309 pair observations (0.66%). A single snapshot cannot show whether the same situation persists across time. In the 58-minute backfill, 24,016 of 4,156,014 pair observations were positive (0.58%). Grouping observations of the same aircraft pair into a visit, with a new visit after a gap of more than 60 seconds, gives 2,200 visits across 2,078 positive aircraft pairs. The 24,016 rows therefore are not 24,016 independent situations. Visits are still rule-based groupings, not verified incidents.

The reported imbalance changes with the unit counted: 0.58% of pair observations, 2.33% of pair visits, and 2.66% of aircraft pairs were positive. About 79% of aircraft that appeared in at least one pair had a positive pair at some point in the hour. That does not mean 79% were in danger; it shows how broadly this proximity rule can apply over time. The [class-rate figure](experiments/figures/step8_rates_by_level.png) makes this change in unit clear: rates rise as repeated observations are grouped into visits, trajectories, and aircraft.

## What distinguishes risky from safe pairs?

Lateral distance and vertical separation distinguish the classes because they define the label. Other motion features provide less separation: in the risky-versus-near-miss comparison, closing speed, bearing difference, and convergence were close to 0.5 AUROC. About half of the positive pairs were converging, similar to the safe groups. The [feature distributions](experiments/figures/step7_feature_distributions.png) and [box plots](experiments/figures/step7_feature_boxplots.png) show that risky and near-miss pairs tend to be lower and slower than other safe traffic, but their motion-feature distributions overlap. The [separation and CPA plots](experiments/figures/step7_feature_scatter.png) show why strong model scores need care: the label-defining distances nearly encode the answer by construction.

The [representative trajectory plots](experiments/figures/step4_trajectories.png) and [geographic plot](experiments/figures/step4_geography.png) help inspect example paths and where positive observations occurred. They do not yet provide a systematic count of crossing, overtaking, or head-on encounters. Likewise, the [regional rate map](experiments/figures/step8_rates_by_region.png) shows that some less-busy cells have higher positive rates than the busiest Los Angeles cell, but these rates include repeated observations of the same pairs. Airport proximity has not yet been checked against an authoritative airport dataset.

## Duration and time patterns

The duration analysis covers all observed aircraft tracks in the backfill, not risky tracks versus safe tracks. It found 873 track segments and a censoring-aware median observed duration of 1,410 seconds. This is time under observation, not full flight duration. The [survival plot](experiments/figures/step9_trajectory_duration_survival.png) accounts for tracks whose collection ended before their observed segment was known to end. A direct risky-versus-safe duration comparison remains to be done.

Across the four smaples that i have collected on one day, positive pair-observation rates were 0.390% at 6:39 AM, 0.842% at 10:18 AM, 0.787% at 2:30 PM, and 0.585% at 7:39 PM. The [time-of-day plot](experiments/figures/step10_time_of_day_rates.png) shows the highest rate in the 10:18 AM sample and the lowest in the 6:39 AM sample. With only one sample at each time on one date, this is descriptive—not evidence of a repeating daily or seasonal pattern. Traffic mix and coverage may also differ between sessions.

## Takeaway

The backfill answers part of the “55 positives” question: repeated rows collapse into far fewer pair visits, and the positive class does not consistently show a distinctive convergence or heading pattern under the current label. But these are proximity labels, not confirmed safety events. The duration comparison by class, encounter-geometry taxonomy, airport validation, and multi-date time analysis remain open.

For the next report, the most useful additions are repeated collections across dates, a direct duration comparison for risky-involved and safe tracks, and systematic geometry categories based on relative motion. That evidence should come before a broad effort to optimize the GNN or its imbalance strategy.

**********************************************************************************************************

## Phase 1 — Existing Data and Current Labels
.

### Step 2: Current labels

For each snapshot, a pair is positive when lateral distance is **≤5 nautical miles** and vertical separation is **≤1,000 feet**; otherwise it is negative if both values are available. Missing values remain unlabelled and are dropped before modeling. This is a rule-based current-proximity label, not a future conflict or confirmed near-collision. Distance and vertical separation are also model inputs, so they directly encode the label rule.

## Summarize for step 3

I reviewed the project’s logistic-regression comparison. The clearest results are from the backfill-only temporal validation; they are validation results, not final test-set results. The project notes say the later multi-session split is separate, and its test set was held out for subsequent comparisons. README.md · step6_split_comparison.csv

Plain logistic regression, at the default 0.5 threshold: precision 0.837, recall 0.754, F1 0.794, average precision 0.895, and AUROC 0.9993. It found 3,300 of 4,376 validation positives, missing 1,076.
SMOTE logistic regression: precision 0.497, recall 1.000, F1 0.664, average precision 0.900, and AUROC 0.9993. It found all 4,376 positives, with more false alarms. SMOTE was selected under the project’s rule prioritizing fewer missed positives. week3_imbalance_comparison.csv · week3_imbalance_selection.md

Interpretation: The high AUROC and average precision do not show that the model predicts future conflicts. The current label is determined by same-time lateral distance and vertical separation, which are also model inputs. The project reports that removing those two features sharply reduced AUROC to about 0.75 and average precision to about 0.018—near the roughly 0.6% positive rate. That supports the concern that the model is largely reconstructing the current label rule. README.md

## step 4

Yes. I checked the model-eligible pairs against the saved split: 510 distinct aircraft pairs appear in both training and test—about 2.67% of the test pairs.

## Step 5
Reuse the ADS-B cleaning, pair-feature pipeline, existing logistic-regression and GraphSAGE models, and evaluation tools...

change the current snapshot-based labels to validated future-conflict onset labels, prevent leakage and address pair overlap, and add candidate-recall, ranking, lead-time, and end-to-end scaling evaluations.

## Phase 1 Summary

The project builds aircraft-pair features from Southern California ADS-B snapshots and compares logistic regression with GraphSAGE using a rule-based label: separation is at most 5 NM horizontally and 1,000 ft vertically. In the current multi-session test results, logistic regression outperformed GraphSAGE on average precision (0.933 vs. 0.895); these scores measure the current proximity label, not future-conflict prediction. The main limitations are that the label-defining distances are also model inputs, the collected sessions cover only two dates, and 510 aircraft pairs (2.67% of test pairs) occur in both train and test. Reuse the data-cleaning, feature, model, and evaluation pipelines; define and validate a future-onset target, handle tracking gaps and pair overlap to prevent leakage, and measure candidate-generation recall, ranking, lead time, and end-to-end graph scaling.

## Phase 2 Summary — Draft Future-Conflict Target

**Provisional; confirm with the professor before using in experiments.**

### Target definition

At each snapshot time \(t\), predict whether an eligible aircraft pair within the existing 50 NM candidate radius will enter the current proximity limits—lateral separation ≤5 NM and vertical separation ≤1,000 ft—during the next 60 seconds. Exclude pairs already within both limits at \(t\), and use only information available at \(t\) as model inputs.

### Label-generation method

Label a pair positive if a later observed paired position enters both limits during \((t, t+60\text{ s}]\). Label it negative only if valid paired observations cover the full horizon, with no gap over 30 seconds, and no threshold entry is observed. If no positive is observed and follow-up is incomplete or a gap exceeds 30 seconds, mark the example unknown and exclude it. Future observations are used only to assign labels; these are rule-defined proximity events, not confirmed collisions.

## Phase 3, Step 1 — Cleaned Input Data

Created cleaned copies of the five modeling sessions and a cleaning audit; the raw files were left unchanged. The cleaner removes ground, stale, missing/unparseable, out-of-range, and duplicate-per-aircraft-per-snapshot records, while retaining and flagging repeated position reports across distinct snapshots. It retained **219,016 of 280,961 rows**; no out-of-range coordinates or kinematics, or duplicate aircraft/snapshot records, were found. See [`data/processed/state_cleaning_audit.csv`](data/processed/state_cleaning_audit.csv) and [`src/data/clean_states.py`](src/data/clean_states.py).

The sessions had usable timestamps and a median sampling interval of about 10 seconds; one 10:18 AM session contained a 284-second gap. Across the five sessions, 228 rows had missing or unparseable required values, and 41,179 had stale or inconsistent position times and were filtered out. No invalid coordinates, speed/track values, or ICAO24 identifiers were found. There were no exact duplicate rows or duplicate aircraft records within a snapshot; 1,814 repeated position reports across different snapshots were retained and flagged.

## Phase 3, Step 2 — Sort by Aircraft and Time

Within each session, cleaned observations are now sorted by aircraft identifier (`icao24`) and then snapshot timestamp, so each aircraft’s records are in chronological order. Sessions remain separate, and repeated position reports remain flagged rather than being removed.

**Observations:** The cleaned data is now ordered chronologically for each aircraft within each session, making it easier to track aircraft over time. Repeated position reports still occur, so they are retained and flagged rather than treated as new movement.

## Phase 3, Step 3 — Future Labels Without Future Features

Implemented a separate future-labeling pipeline. Each model-input row contains pair features from prediction time only; later observations supply the `future_conflict` target and label status, but are not written as features. A positive requires a later threshold entry with at least one new aircraft position report; negatives require paired tracking through the horizon with no gap over 30 seconds. Ongoing conflicts and insufficient follow-up are excluded from model labels, not treated as negatives. The default 60-second horizon and 5 NM / 1,000 ft thresholds remain provisional pending professor confirmation. The labeling code is in [`src/labeling/future_conflicts.py`](src/labeling/future_conflicts.py); generated session datasets and an audit are written under `data/processed/future_labels/`.

Across the five sessions, **8,688,628 pair-snapshot rows** were labeled: 20,224 positive future-onset rows and 7,524,300 negatives. The other rows were excluded as ongoing conflicts (58,244), unknown current separation (4,103), or incomplete follow-up (1,081,757). These are repeated pair-time observations, not distinct conflict events. `prepare_future_model_inputs` returns only the six prediction-time features, with the target and label status kept separate; validation on a saved session confirmed neither label column enters the feature matrix.

**Observation:** Most labeled rows were negative, while many were excluded because the pair was already within the limits or did not have enough follow-up data. The model inputs contain only information from prediction time; later observations were used only to assign labels.

## Phase 3, Step 5 — Training, Validation, and Test Splits

Assigned whole collection sessions using the existing chronological split: the September 25 backfill and October 7 morning sessions for training, the 2:30 PM session for validation, and the 7:39 PM session for testing. This keeps snapshots from the same session together and tests on later data; it does not duplicate the large labeled files. The manifest is [`data/processed/future_labels/future_label_split_manifest.csv`](data/processed/future_labels/future_label_split_manifest.csv), and [`src/evaluation/future_label_splits.py`](src/evaluation/future_label_splits.py) provides a loader that excludes unknown labels.

The split contains 6,572,083 training rows (15,450 positive), 1,353,379 validation rows (3,197 positive), and 763,166 test rows (1,577 positive), before removing ongoing/unknown examples. **443 distinct aircraft-pair identities (2.52% of test pairs) recur in both train and test**, though their observations are from separate sessions; pair identifiers are not model features. The split evaluates later-session performance, not generalization to entirely unseen pairs, aircraft, dates, or seasons.

**Observation:** Keeping whole sessions in separate splits lets us test the model on later recordings without copying the large data files. Some aircraft pairs still recur across splits, so this does not test performance on entirely unseen pairs.

## Phase 3, Step 6 — Training-Only Preprocessing

Added a streaming helper that fits feature scaling using only known, usable training examples and saves the scaler; validation and test rows are not used to calculate scaling statistics. The six model features are predefined in the existing feature helper, so no data-driven feature selection has been performed.

**Observation:** Scaling statistics come from training data only, avoiding information from validation or test data leaking into preprocessing. The feature list remains fixed rather than being selected from the data.

## Phase 3, Step 7 — Positive Encounter Availability

Grouping positive forecast rows for the same aircraft pair into one episode when consecutive positive snapshots are no more than 30 seconds apart gives **4,186 positive episodes**: 3,104 training, 727 validation, and 355 test. This supports a preliminary evaluation, but the test positives come from one session on one day, so the results will not establish broad generalization; these grouped forecast episodes are an estimate, not verified real-world incidents.

**Observation:** There are 355 estimated positive episodes in the test set, enough for an initial evaluation, but the single test session limits how broadly the results can be generalized.

## Phase 3, Step 8 — Save Data and Reproduction Code

The cleaned session files and cleaning audit are saved under `data/processed/`; future-label CSVs, their audit, the split manifest, and the fitted training-only scaler are saved under `data/processed/future_labels/`. The generation code and default parameters are in [`src/data/clean_states.py`](src/data/clean_states.py), [`src/labeling/future_conflicts.py`](src/labeling/future_conflicts.py), and [`src/evaluation/future_label_splits.py`](src/evaluation/future_label_splits.py). These generated data and scaler files are Git-ignored because they are large; keep a copy separately if you need the exact processed results, and retain access to the raw input sessions to regenerate them.

To recreate the outputs, run `python -m src.data.clean_states`, then `python -m src.labeling.future_conflicts`, then `python -m src.evaluation.future_label_splits`. The audit files record the cleaning and label settings used.

## Phases 1–3 in Simple Words

- **Phase 1 — Understand the existing project:** Reviewed the ADS-B data, current proximity labels, and the logistic-regression and GraphSAGE results. The key finding was that the current label is directly defined by distance features the models also receive, so strong scores do not prove that they can predict future conflicts.
- **Phase 2 — Decide what future event to predict:** Drafted a rule for predicting whether an aircraft pair will enter the proximity limits within 60 seconds. Future observations are used only to create the label, not as prediction-time inputs. The thresholds and tracking-gap rules are provisional and should be confirmed with the professor.
- **Phase 3 — Prepare a reproducible dataset:** Cleaned the observations, generated future-conflict labels, assigned whole sessions to training, validation, and test, and fitted feature scaling using training data only. Documented the settings and how to recreate the files. This phase prepares the data; training and comparing the future-conflict models is still to be done.

## Phase 4, Step 1 — Aircraft as Graph Nodes

`build_nodes` in [`src/graph/sparse_graph.py`](src/graph/sparse_graph.py) turns one snapshot into one node per usable airborne aircraft, using the existing cleaning rules. Each node carries latitude, longitude, altitude (barometric, else geometric), speed, and heading as sine/cosine; row position is the node index. Only that snapshot's reported state is used, so no future information enters the nodes. Checked on a real 7:39 PM snapshot: 169 aircraft gave 169 unique nodes.

Simple words: In Step 1, I made each aircraft in a snapshot into one "dot" (a node) in a graph. Each dot stores what we know about that plane at that moment: its position, altitude, speed, and direction.

One snapshot at a time: the function only looks at one moment in time. It never uses what happens later, so nothing from the future leaks in.
Unusable aircraft are skipped: planes on the ground or with bad data are left out, using the same cleaning rules as Phase 3.
Tested: I checked it on small test cases and on a real snapshot, where 169 aircraft became 169 dots.

## Phase 4, Step 2 — Candidate-Edge Rule

**Rule (provisional):** connect two aircraft when their current lateral distance is at most **15 NM** (`CANDIDATE_RADIUS_NM` in [`src/graph/sparse_graph.py`](src/graph/sparse_graph.py)). A fixed-radius rule matches the 5 NM conflict threshold and the physical limit on how far aircraft can close in 60 seconds, whereas a fixed neighbor count would link distant aircraft in sparse airspace and drop close ones in dense airspace.

**Evidence:** Across the 20,224 positive rows in the five sessions, the farthest pair at prediction time was 19.8 NM apart, and the 99th percentile was 9.4 NM. Compared with the previous 50 NM pair radius:

| Radius | Share of future-conflict positives inside | Share of all candidate pairs kept |
|---|---|---|
| 10 NM | 99.38% | 7.7% |
| **15 NM** | **99.91%** | **15.6%** |
| 20 NM | 100.00% | 25.1% |
| 30 NM | 100.00% | 48.4% |

15 NM keeps nearly all conflicts while removing about 84% of pairs. Use 20 NM if complete recall matters more than sparsity. These figures are measured on this dataset and the provisional 60-second label, so they should be rechecked if the label changes; the full recall report is Step 8.

Simple words: In Step 2, I picked the rule for connecting aircraft in the graph. Two planes get a line (an edge) between them only if they are within 15 nautical miles of each other.

Observations:
- Conflicts start close: almost every future conflict (99.91%) began with the two planes less than 15 NM apart. The farthest was 19.8 NM.
- Most pairs are irrelevant: a 15 NM radius removes about 84% of all possible pairs, so the graph stays small and fast.
- Trade-off: a bigger radius, such as 20 NM, catches every conflict but keeps more pairs. I chose 15 NM as a balance.
- Why a radius: a fixed number of neighbors would link far-apart planes in quiet airspace and miss close ones in busy airspace.

This is provisional, because it depends on the 60-second label chosen in Phase 2. Nothing is applied yet; Step 6 will use the rule to build the edges.

## Phase 4, Step 3 — Edge Direction Convention

**Decision:** edges are **undirected**. Each candidate pair is stored once, with the lower node index first, and the GraphSAGE message-passing step reads it in both directions so both aircraft receive information from the other. The conflict label is also about the pair, not one aircraft acting on the other, so a pair has one score regardless of order. This matches how the existing GraphSAGE code builds its graphs (one stored pair per edge, both directions for message passing). The convention is recorded next to `CANDIDATE_RADIUS_NM` in [`src/graph/sparse_graph.py`](src/graph/sparse_graph.py) and will be applied when the edges are built in Step 6.

Simple words: In Step 3, I decided a connection between two aircraft has no direction. "A–B" and "B–A" are the same line, stored once, and both planes learn from each other.

Observations:
- A conflict involves both aircraft equally, so a one-way edge would not describe it well.
- Storing each pair once avoids scoring or counting the same pair twice, and gives half as many stored edges as a directed graph would.
- Nothing is built yet; this is a rule that Step 6 follows.

### Step 3 in Simple Words

**What I did:** I decided how to treat the connection (edge) between two aircraft. An edge can be directed (A→B differs from B→A) or undirected (A–B is one link). I chose **undirected**.

- A conflict involves both aircraft equally, so "A near B" and "B near A" are the same fact.
- Each pair is stored **once**, with the lower aircraft number first, so there are no duplicates.
- During GraphSAGE message passing, information flows **both ways**: A learns from B, and B learns from A.
- Each pair gets **one score** (for example 0.91 for A–F), which is what the ranking table needs.

**What I observe:**
- Undirected storage halves the stored edge count compared with a directed graph, which helps the sparse-graph speed argument.
- It matches the existing GraphSAGE code, so no code change is needed.
- This was a decision, not a computation. The only code change is a comment in [`src/graph/sparse_graph.py`](src/graph/sparse_graph.py); Step 6 applies the rule when it builds the edges.

## Phase 4, Step 4 — Prediction-Time Node and Edge Features

**Node features** (from Step 1, `build_nodes`): latitude, longitude, altitude (m), speed (m/s), and heading as sin/cos. Each comes from that aircraft's current reported state.

**Edge features** (`build_edge_features` in [`src/graph/sparse_graph.py`](src/graph/sparse_graph.py)): one row per candidate edge, computed from the two nodes' current state only: lateral distance (NM), vertical separation (m), closing speed (m/s), bearing difference (deg), log time to closest approach, and a has-closest-approach flag. These are the same six features, formulas and transforms as the pairwise baseline (`MODEL_FEATURES`), so the baseline and GraphSAGE see identical pair information. A test checks that the graph edge features equal the baseline pipeline's values for the same pair.

Simple words: In Step 4, I decided what facts the model gets. Each plane (dot) carries its own position, height, speed and heading. Each connection (line) carries facts about the two planes together: how far apart they are, how fast they are closing, and when they would be closest.

Observations:
- The edge features are the same as the baseline's, so any GraphSAGE gain comes from the surrounding traffic, not from extra pair information.
- Distance and vertical separation are still included, so the baseline can still partly rebuild the label rule. Dropping them is an open choice for the later model comparison.
- Only current-snapshot data is used; no future information. Step 5 documents this check.
## Phase 4, Step 5 — Prediction-Time Information Only

**Rule:** a graph is built from one snapshot at time *t*, and nothing else. `build_nodes` rejects input with more than one `snapshot_time`, and node and edge features (`NODE_COLUMNS`, `EDGE_FEATURE_COLUMNS`) are computed only from that snapshot's reported positions, altitudes, speeds and headings. The labels (`future_conflict`, `future_conflict_status`) are never graph inputs; they are attached separately to the candidate edges for training and evaluation only. The candidate rule (Step 2) uses current distance only. A test in `tests/test_sparse_graph.py` checks that label columns do not change the graph, that they are not feature columns, and that mixed-time input is rejected.

Simple words: In Step 5, I made sure the graph only knows what a controller could know right now. It never sees what happens in the next 60 seconds. The answer key (the label) is kept separate and is used only to grade the model.

Observations:
- The leak protection is built into the code (one snapshot only, label columns not used), not just promised.
- The 15 NM radius was chosen by looking at future conflicts, but it is a fixed rule applied to current distance only, so it does not use future data when the graph is built. If the radius is later tuned, use training sessions only.

## Phase 4, Step 6 — Efficient Neighbor Search

**Method:** `build_candidate_edges` in [`src/graph/sparse_graph.py`](src/graph/sparse_graph.py) builds a scikit-learn `BallTree` on great-circle (haversine) distance and asks it for each aircraft's neighbours within `CANDIDATE_RADIUS_NM` (15 NM). It never builds the all-pairs distance table, which is what makes the old method O(N²). Pairs are returned once with the lower index first (Step 3). Edge features (Step 4) are then computed only for these candidate edges.

**Checks:** a test on 300 random aircraft confirms the edge set is exactly the same as a brute-force all-pairs check, and that edges are undirected.

**Timing** (synthetic aircraft spread uniformly over a 3° × 4° box; this measures the method, not real traffic; the first row includes start-up cost):

| Aircraft N | All possible pairs | Candidate edges (≤ 15 NM) | Tree search (ms) | Brute-force distance of all pairs (ms) |
|---|---|---|---|---|
| 100 | 4,950 | 84 | 32.6 | 2.0 |
| 250 | 31,125 | 610 | 3.5 | 6.6 |
| 500 | 124,750 | 2,221 | 6.1 | 15.5 |
| 1,000 | 499,500 | 9,114 | 10.0 | 74.6 |
| 5,000 | 12,497,500 | 228,304 | 105.1 | 3,066.9 |

Simple words: In Step 6, I changed how we find nearby planes. Instead of measuring the distance between every possible pair of planes, the code uses a spatial lookup that only checks planes that are actually close, like looking at the neighbours on your street instead of calling every house in the city.

Observations:
- With 5,000 aircraft, the tree search took about 0.1 s versus about 3 s for just the all-pairs distance step, and the gap grows as N grows.
- At N = 100 brute force is faster, because the tree has a fixed setup cost. The tree pays off at larger N.
- The number of edges grows roughly with N times the average neighbours, not N², which supports the sparse-graph argument. Real traffic is clustered around airports, so real degree will be higher; Step 7 measures it on the data.

## Phase 4, Step 7 — Counts and Average Degree per Snapshot

**What is recorded:** `build_graph` in [`src/graph/sparse_graph.py`](src/graph/sparse_graph.py) builds the whole graph for one snapshot (nodes, candidate edges, edge features) and returns its counts. `snapshot_graph_counts` repeats it for every snapshot, recording aircraft, all possible pairs, candidate edges, and average degree (2 × edges ÷ aircraft, since each undirected edge touches two aircraft). Run `python -m src.graph.sparse_graph` to write one CSV per session to `data/processed/graph_stats/` (Git-ignored, regenerated from `cleaned_states/`).

| Session | Snapshots | Mean aircraft | Mean all possible pairs | Mean candidate edges (max) | Mean average degree (max) |
|---|---|---|---|---|---|
| backfill 1790375809 | 349 | 265.7 | 35,192 | 1,754 (2,037) | 13.2 (14.7) |
| session 06:39 AM | 167 | 103.0 | 5,339 | 238 (445) | 4.4 (7.0) |
| session 10:18 AM | 145 | 289.7 | 41,865 | 2,369 (2,661) | 16.4 (18.7) |
| session 02:30 PM | 168 | 236.6 | 27,894 | 1,356 (1,669) | 11.4 (13.3) |
| session 07:39 PM | 169 | 161.7 | 13,012 | 762 (996) | 9.4 (11.6) |
| **All** | **998** | **219.5** | | **1,354** | **11.3** (median 12.4, max 18.7) |

Simple words: In Step 7, I counted how big the graph is in every snapshot: how many planes, how many connections, and how many connections each plane has on average.

Observations:
- Across all 998 snapshots, the 15 NM rule keeps only about 5.2% of all possible pairs.
- The average plane has about 11 neighbours (maximum 18.7 in any snapshot), which is close to the "about 10 neighbours" in the professor's Ho example(in the email). The graph size grows with the number of aircraft times neighbours, not with N².
- Busier sessions have a higher degree (10:18 AM: 16.4; 06:39 AM: 4.4), so the saving is smaller in busy airspace but still large.
- These counts are over the full cleaned data, not only the labeled rows, so they include the pre-conflict and ongoing pairs that Phase 3 excluded from labels.

## Phase 4, Step 8 — Candidate-Generation Recall

**What is measured:** for every positive (future-conflict) pair, the graph is rebuilt from the cleaned states of that snapshot exactly as at prediction time (`build_nodes`, then `build_candidate_edges` at 15 NM), and the pair is checked for membership among the edges. Code: [`src/graph/candidate_recall.py`](src/graph/candidate_recall.py); run `python -m src.graph.candidate_recall` (writes `data/processed/graph_stats/candidate_recall.csv`, Git-ignored). Episodes are the Phase 3 Step 7 groups (new episode after a gap above 30 s). Two episode measures are reported: **any row** (the episode is flagged at least once before the conflict) and **first row** (it is flagged at its earliest warning time).

| Split | Positive rows | Rows in candidates | Row recall | Episodes | Episode recall (any row) | Episode recall (first row) |
|---|---|---|---|---|---|---|
| Train | 15,450 | 15,433 | 99.89% | 3,104 | 100% | 99.65% (3,093) |
| Validation | 3,197 | 3,195 | 99.94% | 727 | 100% | 99.72% (725) |
| Test | 1,577 | 1,577 | 100% | 355 | 100% | 100% |
| **All** | **20,224** | **20,205** | **99.91%** | **4,186** | **100%** | **99.69% (4,173)** |

Simple words: In Step 8, I checked whether the graph throws away real conflicts. If a conflict pair is not connected in the graph, the model never gets to score it, so this number is the upper limit on what the model can catch.

Observations:
- The 15 NM graph keeps 99.91% of future-conflict rows (19 of 20,224 are missed), and every one of the 4,186 conflict episodes appears as an edge at least once before the conflict.
- At the very first warning moment, 13 episodes (0.31%) were still outside 15 NM, but they came within range in a later snapshot.
- The test session has 100% recall. The radius was chosen by looking at all sessions together, so the test figure is not fully independent; if the radius is retuned, use the training sessions only.
- Limitation: the Phase 3 labels only contain pairs within 50 NM, so a conflict pair that was farther than 50 NM at prediction time is outside the labeled data and not counted here.

## Phase 4 Deliverable — Graph Construction and Report

**Graph-construction function:** `build_graph(snapshot)` in [`src/graph/sparse_graph.py`](src/graph/sparse_graph.py). For one snapshot it returns the nodes (Step 1), undirected candidate edges within 15 NM (Steps 2, 3, 6), edge features (Step 4), and the counts (Step 7). It uses only that snapshot's information (Step 5).

| Item | Result |
|---|---|
| Construction method | BallTree (haversine) radius search; no all-pairs distance table. Matches brute force exactly (tested). 5,000 aircraft: about 0.1 s versus about 3 s for the all-pairs distance step alone. |
| Candidate rule | Lateral distance of at most 15 NM, undirected, each pair stored once |
| Candidate counts (998 snapshots) | Mean 219.5 aircraft and 1,354 candidate edges per snapshot (maximum 2,661 edges) |
| Graph density | Candidate edges are about 5.2% of all possible pairs; average degree 11.3 (median 12.4, maximum 18.7) |
| Candidate-generation recall | 99.91% of future-conflict rows (20,205 of 20,224); 100% of the 4,186 conflict episodes appear as an edge at least once; 99.69% at their first warning time. Test split: 100% |

Simple words: The graph now keeps about 1 in 20 possible pairs, builds without checking every pair, and still keeps almost every real future conflict, so the model has few missed conflicts to worry about.

Observations:
- Sparse and complete enough: the graph is about 20 times smaller than all pairs while losing under 0.1% of conflict rows.
- The 15 NM radius, 60-second horizon and the label thresholds are provisional until the professor confirms them. Rerun `python -m src.graph.sparse_graph` and `python -m src.graph.candidate_recall` if they change.

## Phase 5, Step 1 — Logistic Regression Trained on the Future-Conflict Label

**What changed:** the baseline is still plain logistic regression (scikit-learn `StandardScaler` followed by `LogisticRegression(max_iter=5000, random_state=42)`, no class weighting or SMOTE), but its target is now `future_conflict` from Phase 3 instead of the current-proximity label. Code: [`src/baselines/future_logistic_regression.py`](src/baselines/future_logistic_regression.py); run `python -m src.baselines.future_logistic_regression`. The model is saved to `models/future_logistic_regression.joblib` (Git-ignored).

**Training data:** known-label pairs from the training sessions only, kept only if they are Phase 4 candidate edges (lateral distance of at most 15 NM). That gives 828,423 pairs, of which 15,433 are positive (1.86%). The 17 training positives outside 15 NM are not available to either model. The scaler is fitted inside the pipeline on these training rows only.

Simple words: In Step 1, I kept the simple model but changed what it learns. It now learns to predict whether two planes will get too close in the next 60 seconds, not whether they are too close right now. It only studies the training sessions, and only pairs that would be connected in the graph.

Observations:
- The training set is small compared with the 5.7 million rows in the Phase 3 training split, because the 15 NM rule keeps only the nearby pairs, and the positive rate rises to 1.86%.
- No test or validation data was used. Predictions, thresholds and evaluation come in the next steps.

## Phase 5, Step 2 — Prediction-Time Information Only

**Check:** the trained model's input columns were read back from the saved file, and they are exactly the six pair features in `MODEL_FEATURES`: `lateral_distance_nm`, `vertical_separation_m`, `closing_speed_mps`, `bearing_difference_deg`, `log_time_to_cpa`, `has_cpa`. All six are computed from the two aircraft's reported state at the prediction snapshot, with the same formulas as the GraphSAGE edge features (Phase 4, Step 4). The label (`future_conflict`), its status, and the snapshot and aircraft identifiers are separated out by `prepare_future_model_inputs` and are never inputs. An existing test, `test_future_targets_and_status_are_not_model_features` in `tests/test_future_conflicts.py`, enforces this, and the scaler is fitted on training rows only.

Simple words: In Step 2, I checked that the simple model only sees what is known right now: how far apart two planes are, their height difference, how fast they are closing, their heading difference, and the estimated closest-approach time. It never sees the answer or anything from the next 60 seconds.

Observations:
- The model's six input columns match the expected list exactly, so no label or future column slipped in.
- Distance and vertical separation define the label's thresholds, so this baseline can partly rebuild the labelling rule. The professor's email raised this concern. Dropping them is an open option for the comparison.
- No new code was needed for this step, only the check and this record.

## Phase 5, Step 3 — Same Pairs, Splits and Target as GraphSAGE

**One shared definition:** `load_split(split)` in [`src/baselines/future_logistic_regression.py`](src/baselines/future_logistic_regression.py) defines the evaluation set, and GraphSAGE must score exactly these pairs. A pair is **eligible** when (1) its future-conflict label is known (`positive` or `negative`; ongoing and unknown-follow-up pairs are excluded as in Phase 3), (2) it is a Phase 4 candidate edge (lateral distance of at most 15 NM), and (3) its six features are finite. The target is `future_conflict` (60-second horizon), and the splits are the Phase 3 chronological session splits.

| Split | Sessions | Eligible pairs | Positive | Positive rate |
|---|---|---|---|---|
| Train | backfill, 06:39 AM, 10:18 AM | 828,423 | 15,433 | 1.86% |
| Validation | 02:30 PM | 186,418 | 3,195 | 1.71% |
| Test | 07:39 PM | 105,856 | 1,577 | 1.49% |

**Consistency check:** the positive counts (15,433, 3,195, 1,577) equal the "rows in candidates" counts from the Phase 4 recall measurement, which used the BallTree edges. So the distance filter used here selects the same positives as the graph's candidate edges.

Simple words: In Step 3, I fixed the exam for both models. They are trained, tuned and tested on the same sessions, the same pairs (only those connected in the graph), and the same question (will this pair get too close in 60 seconds?). That way any difference in score comes from the model, not from different data.

Observations:
- The test session has the lowest conflict rate (1.49%), so the test task is slightly harder than training in terms of class balance, and test scores should be read with that in mind.
- The 19 conflict rows that fall outside 15 NM are not eligible for either model. Both models share that limit.
- GraphSAGE should not use `load_split` for its own features, only to decide which pairs to score and with which labels.

## Phase 5, Step 4 — Probability for Each Candidate Pair

**What is produced:** `save_predictions` in [`src/baselines/future_logistic_regression.py`](src/baselines/future_logistic_regression.py) scores every eligible pair of the validation and test splits (the Step 3 set) with `predict_proba` and writes one row per pair: `snapshot_time`, `icao24_a`, `icao24_b`, the true `future_conflict` label, and `probability` (the predicted chance of entering a proximity conflict within 60 seconds). Run `python -m src.baselines.future_logistic_regression`; files are saved to `data/processed/future_predictions/logreg_validation_predictions.csv` and `logreg_test_predictions.csv` (Git-ignored, regenerated by the command). The model was fitted on training data only.

| Split | Pairs scored | Probabilities in 0–1 | Missing | Duplicate pairs | Mean probability | Actual positive rate |
|---|---|---|---|---|---|---|
| Validation | 186,418 | yes | 0 | 0 | 0.0171 | 1.71% |
| Test | 105,856 | yes | 0 | 0 | 0.0151 | 1.49% |

Simple words: In Step 4, the model gave every nearby pair of planes a score between 0 and 1: its estimate of how likely the pair is to get too close within 60 seconds. These scores are saved so they can be ranked, thresholded and compared with GraphSAGE.

Observations:
- Every eligible pair got exactly one probability, with no gaps or duplicates.
- The average predicted probability is almost the same as the real conflict rate (1.71% versus 1.71%, and 1.51% versus 1.49%), so the model's scores are well calibrated on average.
- The probabilities are not yet turned into yes/no decisions. The threshold is chosen from validation data in Step 5, and results are measured afterwards.

## Phase 5, Step 5 — Threshold Chosen on Validation Data Only

**Method:** the threshold is picked by `fit_threshold` in [`src/evaluation/threshold.py`](src/evaluation/threshold.py), the project's existing rule (a missed conflict costs 5 times a false alarm), moved out of `src/gnn/train.py` into a module that does not need PyTorch, so GraphSAGE and the baseline use the same code. It reads only the validation predictions (`logreg_validation_predictions.csv`) and never sees the test set. The chosen value is saved with `"chosen_on": "validation"` in `data/processed/future_predictions/logreg_threshold.json` by `python -m src.baselines.future_logistic_regression`.

**Chosen threshold: 0.1557.** A pair is flagged as a likely conflict when its probability is at least this value. As a sanity check, the same fixed threshold gives:

| Split | Pairs flagged | True conflicts caught | False alarms | Missed | Precision | Recall |
|---|---|---|---|---|---|---|
| Validation (used to choose it) | 6,018 | 2,288 | 3,730 | 907 | 0.380 | 0.716 |
| Test (applied, not tuned) | 2,860 | 1,096 | 1,764 | 481 | 0.383 | 0.695 |

Simple words: In Step 5, I decided how high a score must be before the model raises an alarm: 0.1557. I picked that number using only the validation session. The test session was kept hidden until I applied that fixed number. Missing a real conflict is treated as five times worse than a false alarm, so the number is low enough to catch most conflicts.

Observations:
- Validation and test behave almost the same (precision 0.380 versus 0.383, recall 0.716 versus 0.695), which suggests the threshold is not overfitted to the validation session.
- Roughly 2 of every 3 alarms are false, which is the cost of catching about 70% of conflicts. The full metric set (F1, average precision, Recall@K, false positives per time window) comes in the later evaluation phase.
- The 5:1 cost is a project choice carried over from the earlier phases, not a value from the professor, so it is worth confirming.

## Phase 5, Step 6 — Recorded Features, Preprocessing, Settings and Prediction Time

**Where:** `write_run_record` in [`src/baselines/future_logistic_regression.py`](src/baselines/future_logistic_regression.py) saves everything to `data/processed/future_predictions/logreg_run_record.json` each time `python -m src.baselines.future_logistic_regression` runs (Git-ignored; regenerated by the command). The trained model is `models/future_logistic_regression.joblib`.

| Item | Recorded value |
|---|---|
| Features (6) | `lateral_distance_nm`, `vertical_separation_m`, `closing_speed_mps`, `bearing_difference_deg`, `log_time_to_cpa`, `has_cpa` |
| Feature prep | Time to closest approach capped at 3,600 s, then log(1 + t); `has_cpa` = 1 if a closest approach exists |
| Scaling | `StandardScaler` fitted on the 828,423 training pairs only; the mean and scale of every feature are stored in the record |
| Eligible pairs | Known label (positive or negative) and lateral distance of at most 15 NM |
| Model | scikit-learn `LogisticRegression`: L2 penalty (default), C = 1.0, solver lbfgs, max_iter = 5000, random_state = 42, no class weights, no SMOTE |
| Threshold | 0.1557, chosen on validation only (5:1 miss-to-false-alarm cost) |
| Prediction time | Features are taken from the snapshot at time *t* (current state only) and the model predicts conflict onset within 60 s after *t* |
| Scoring time | Scoring the test set (105,856 pairs in 166 snapshots) took 0.014 s, about 0.08 ms per snapshot (model scoring only, after the pairs were loaded) |

Simple words: In Step 6, I wrote down everything needed to rebuild this model and get the same results: which six facts it uses, how they are prepared and scaled, the model settings, the alarm threshold, and the moment it predicts from (now, looking 60 seconds ahead). I also timed how fast it scores, for the speed comparison later.

Observations:
- Learned weights (on scaled features): vertical separation −14.6 and lateral distance −3.0 are by far the largest. So the baseline relies mostly on current distance and height gap, which are exactly the label's thresholds. This supports the professor's concern and the plan to run a version without them last.
- The scoring time covers only the model step. The professor's inference-time comparison should also include building the pair features, since the sparse graph removes most of that work. That belongs in the later runtime phase.
- Re-running the command reproduces the same model, because the data, splits and `random_state` are fixed.

## Phase 5, Step 6b — Second Version Without Distance and Vertical Separation

**What it is:** the same logistic regression, data, splits, scaler, and validation-only threshold rule, but with `lateral_distance_nm` and `vertical_separation_m` removed. Those two define the label thresholds (5 NM, 1,000 ft), so the main baseline can partly rebuild the rule from them. The variant keeps only `closing_speed_mps`, `bearing_difference_deg`, `log_time_to_cpa` and `has_cpa`. It comes from `feature_names(include_label_features=False)`; both versions are produced by `python -m src.baselines.future_logistic_regression` (variant files end in `_no_distance`, for example `logreg_no_distance_run_record.json`).

**Test results** (each at its own threshold chosen on validation only; 105,856 pairs, 1,577 positive, 1.49% positive rate):

| Model | Features | Threshold | Flagged | Precision | Recall | F1 | Average precision |
|---|---|---|---|---|---|---|---|
| Main baseline | 6 | 0.1557 | 2,860 | 0.383 | 0.695 | 0.494 | 0.439 |
| Without distance and vertical separation | 4 | 0.0417 | 0 | 0.000 | 0.000 | 0.000 | 0.026 |

Validation average precision: 0.476 (main) and 0.034 (variant). A model that guesses at random would score about 0.015 (the positive rate).

Simple words: I took away the two facts that define the label (how close, and how far apart in height) and asked the same simple model to try again. Without them it can barely tell dangerous pairs from safe ones. It scores only a little better than guessing, and at the chosen alarm level it raises almost no alarms.

Observations:
- The main baseline's skill comes almost entirely from current distance and vertical separation. Closing speed, heading difference and closest-approach time alone carry very little signal in this model.
- This is why the professor's concern matters: a strong baseline here is mostly the labelling rule, so the fairer test of GraphSAGE is whether surrounding traffic adds information beyond those two features.
- The four-feature variant is a weak reference, not a competitor. If GraphSAGE is also run without the two features, the comparison of interest is GraphSAGE against this near-chance baseline.
- The variant's threshold (0.0417) flags almost nothing because no probability level separates conflicts well under the 5:1 cost rule, so look at average precision for it.

## Phase 5 Deliverable — Reproducible Logistic Regression Baseline

**Reproduce:** `python -m src.baselines.future_logistic_regression` trains both versions on the training sessions, scores validation and test, picks each threshold on validation only, and writes everything below. Code: [`src/baselines/future_logistic_regression.py`](src/baselines/future_logistic_regression.py) and [`src/evaluation/threshold.py`](src/evaluation/threshold.py). All outputs are under `data/processed/future_predictions/` and `models/` (Git-ignored, regenerated by the command).

| Deliverable item | Where |
|---|---|
| Test predictions (one probability per eligible pair) | `logreg_test_predictions.csv` (main), `logreg_no_distance_test_predictions.csv` |
| Evaluation results | `test_results` and `validation_results` in `logreg_run_record.json` and `logreg_no_distance_run_record.json`, summarized in Step 6b |
| Feature list, preprocessing, settings, prediction time | `logreg_run_record.json` (Step 6) |
| Threshold (validation only) | `logreg_threshold.json` |

Headline test result (main baseline, six features): precision 0.383, recall 0.695, F1 0.494, average precision 0.439 on 105,856 eligible pairs. Remaining metrics from the professor's list (Recall@K, false positives per time window, lead time, runtime scaling) are for the later comparison phase, where GraphSAGE is scored on exactly these pairs.

## Phase 5 Deliverable — Full Report (Separate Write-Up)

**Decision:** the **main baseline is the six-feature logistic regression**, which keeps lateral distance and vertical separation. GraphSAGE receives the same six edge features, so the comparison is fair. The four-feature version (without those two) is a secondary reference that shows how much the baseline depends on them.

**Reproduce:** run `python -m src.baselines.future_logistic_regression` from the project root, after the Phase 3 labels exist. It trains both versions on the training sessions only, scores the validation and test sessions, picks each threshold on validation only, and writes all outputs. Code: [`src/baselines/future_logistic_regression.py`](src/baselines/future_logistic_regression.py) and [`src/evaluation/threshold.py`](src/evaluation/threshold.py). The data, splits and `random_state` (42) are fixed, so re-running gives the same model and predictions. Outputs are under `data/processed/future_predictions/` and `models/` (Git-ignored, regenerated by the command).

**Setup (main baseline)**

| Item | Value |
|---|---|
| Target | `future_conflict`: enters 5 NM and 1,000 ft within 60 s after prediction time (provisional definition) |
| Features (6) | `lateral_distance_nm`, `vertical_separation_m`, `closing_speed_mps`, `bearing_difference_deg`, `log_time_to_cpa`, `has_cpa`, all from the snapshot at prediction time |
| Eligible pairs | Known label and lateral distance of at most 15 NM (the Phase 4 candidate edges), the same pairs GraphSAGE is scored on |
| Splits | Train: backfill, 06:39 AM, 10:18 AM sessions. Validation: 02:30 PM. Test: 07:39 PM |
| Model | `StandardScaler` (fitted on training pairs only) then `LogisticRegression` (C = 1.0, lbfgs, max_iter = 5000, random_state = 42, no class weights, no SMOTE) |
| Threshold | 0.1557, chosen on validation only (a missed conflict costs 5 times a false alarm) |

**Test results** (test session: 105,856 eligible pairs, 1,577 positive, 1.49% positive rate; each model uses its own validation-chosen threshold)

| Model | Pairs flagged | Precision | Recall | F1 | Average precision |
|---|---|---|---|---|---|
| **Main baseline (6 features)** | 2,860 | **0.383** | **0.695** | **0.494** | **0.439** |
| Reference (4 features, no distance) | 0 | 0.000 | 0.000 | 0.000 | 0.026 |

For reference, random guessing gives an average precision of about 0.015. On the validation session the main baseline scored precision 0.380, recall 0.716, F1 0.497 and average precision 0.476, close to its test result.

**Files**

| Item | Where |
|---|---|
| Test predictions (one probability per eligible pair) | `data/processed/future_predictions/logreg_test_predictions.csv` (main), `logreg_no_distance_test_predictions.csv` |
| Validation predictions | `logreg_validation_predictions.csv`, `logreg_no_distance_validation_predictions.csv` |
| Evaluation results, feature list, preprocessing, settings, prediction time | `logreg_run_record.json` (main), `logreg_no_distance_run_record.json` |
| Threshold | `logreg_threshold.json`, `logreg_no_distance_threshold.json` |
| Trained models | `models/future_logistic_regression.joblib`, `models/future_logistic_regression_no_distance.joblib` |

Simple words: I retrained the simple model to predict whether two planes will get too close in the next 60 seconds. It is tested fairly, on the same pairs and sessions that GraphSAGE will use. On the test session it catches about 70% of the future conflicts, and about 38% of its alarms are real. Everything needed to rebuild it is saved, and one command reproduces it.

Observations:
- The baseline's skill comes almost entirely from current distance and vertical separation. Without them the model is close to guessing. This matches the professor's point that logistic regression can largely rebuild the labelling rule.
- The fair question for GraphSAGE is therefore whether surrounding traffic adds information beyond those two features, not only whether it beats 0.439 average precision.
- Remaining metrics from the professor's list (Recall@K, false positives per time window, lead time, inference-time and runtime scaling) belong to the later comparison phase, where GraphSAGE is scored on exactly these pairs.
- The 60-second horizon, 5 NM / 1,000 ft thresholds, 15 NM radius and 5:1 cost are provisional project choices, still to be confirmed with the professor.

## Phase 6, Step 1 — Provide GraphSAGE the Sparse Graph

**Implementation:** `sparse_snapshot_to_graph` in [`src/gnn/data.py`](src/gnn/data.py) converts one prediction-time snapshot to the PyTorch Geometric `Data` object used by [`src/gnn/model.py`](src/gnn/model.py). It uses the Phase 4 15 NM BallTree candidate edges, the six current-state node features (latitude, longitude, altitude, speed, and heading sin/cos), and the six current pair features. Each undirected candidate pair is stored once in `edge_label_index` for scoring and in both directions in `edge_index` for GraphSAGE message passing. No label or future data is included.

Aircraft without a finite node feature (for example, no reported altitude) and candidate pairs without finite pair features are omitted, matching the usable-feature requirement of the logistic-regression comparison. The function provides the graph inputs only; future-conflict labels are added separately for training in a later step.

Simple words: In Step 1, I connected the graph we already built to GraphSAGE. GraphSAGE now receives each plane's current position, altitude, speed and heading, plus links to nearby planes and facts about each linked pair. The links are sparse, so far-away pairs are not included.

Observations:
- GraphSAGE receives one graph per snapshot, with every candidate link available in both directions for information sharing.
- Pair labels are deliberately absent at this step, so future outcomes cannot leak into the model inputs.
- A focused test checks the node features, candidate-pair indices, two-way message-passing edges and pair-feature dimensions. Run `python -m unittest tests.test_gnn` when the PyTorch and PyTorch Geometric dependencies are installed.

## Phase 6, Step 2 — Contextualized Aircraft Embeddings

The existing `GraphSAGEEdgeClassifier.encode` method in [`src/gnn/model.py`](src/gnn/model.py) passes each node's features through its GraphSAGE convolution layers, using the sparse graph's `edge_index`. Each layer combines an aircraft's current representation with aggregated representations from its direct neighbors; stacking layers lets information travel farther through the graph. The output is one contextualized embedding per aircraft. This step uses the graph from `sparse_snapshot_to_graph` and adds no new model component.

Simple words: GraphSAGE starts with what is known about each plane. It then lets each plane gather information from the nearby planes connected to it. The result is a richer summary for each plane that includes some of its neighbors' situation.

Observations:
- The model already had the GraphSAGE encoder, so Step 2 reuses it rather than adding duplicate code.
- A focused test checks that encoding returns one finite embedding per node and that using neighbor edges changes the embeddings compared with encoding without edges. Run `python -m unittest tests.test_gnn` when PyTorch and PyTorch Geometric are available.

## Phase 6, Step 3 — Candidate-Pair Representations

The existing `pair_representations` method in [`src/gnn/model.py`](src/gnn/model.py) creates one representation for each candidate pair in `edge_label_index`. It combines the two aircraft embeddings symmetrically—sum, absolute difference, and elementwise product—then appends the six current pair features from `pair_attr` (distance, vertical separation, closing speed, bearing difference, log time to closest approach, and its availability flag). Thus swapping the aircraft order does not change the pair representation, and every candidate edge has one representation.

Simple words: GraphSAGE first makes a neighbor-aware summary for each plane. For every candidate pair, the model puts the two summaries together, adds the facts about their current separation and movement, and makes one combined description of that pair.

Observations:
- This pair-combination code already existed in the GraphSAGE model, so I reused it without adding another implementation.
- A focused test checks that there is one representation per candidate pair, that its size includes both endpoint summaries and pair features, and that the pair features are included unchanged. Run `python -m unittest tests.test_gnn` when PyTorch and PyTorch Geometric are available.

## Phase 6, Step 4 — Future-Conflict Edge Classification

The existing `GraphSAGEEdgeClassifier` in [`src/gnn/model.py`](src/gnn/model.py) turns each candidate pair representation into one logit. `sparse_snapshot_to_graph(snapshot, future_labels)` in [`src/gnn/data.py`](src/gnn/data.py) now aligns known `future_conflict` labels with candidate pairs for that snapshot. The full sparse candidate graph remains available for message passing; only pairs with known positive/negative labels are included in `edge_label_index`, `pair_attr`, and `edge_label` for supervised classification. Unknown or unlabeled candidate edges can still provide graph context but are not used as training targets. Sigmoid converts classifier logits to probabilities; training and the probability evaluation pipeline come in subsequent steps.

Simple words: In Step 4, GraphSAGE gets one answer for each labeled nearby pair: how likely the pair is to enter a conflict within the next 60 seconds. It can still learn context from nearby aircraft whose future answer is unknown, but it does not train on those unknown answers.

Observations:
- Labels are matched to candidate edges by the two aircraft identifiers, regardless of pair order, and must belong to the same snapshot.
- The classifier returns one logit per labeled candidate pair; applying sigmoid gives a probability from 0 to 1.
- A focused test checks label alignment and probability output. Run `python -m unittest tests.test_gnn` when PyTorch and PyTorch Geometric are available.

## Phase 6, Step 5 — Tune Model Settings on Validation Only

The small tuning runner is [`src/gnn/future_tuning.py`](src/gnn/future_tuning.py). Run `python -m src.gnn.future_tuning`. It trains each configuration on the training sessions and selects the best by **validation average precision**. It does not load or inspect the test split. Scaling statistics are fitted on training graphs only. The saved tuning history and selected configuration go to `data/processed/graphsage_tuning/` (Git-ignored).

| Hidden channels | Learning rate | Best epoch (out of 5) | Validation average precision |
|---:|---:|---:|---:|
| 32 | 0.0010 | 5 | 0.2356 |
| 32 | 0.0005 | 4 | 0.2160 |
| **64** | **0.0010** | **5** | **0.2550** |
| 64 | 0.0005 | 5 | 0.2453 |

**Selected settings for Step 6:** 64 hidden channels, learning rate 0.001, 2 GraphSAGE layers, dropout 0.2, mean aggregation, and 5 epochs. The final settings were selected only by validation average precision (0.2550). The selection run used weighted binary cross-entropy to account for the rare positive labels; that training procedure was held constant across configurations.

**Fairness check:** the tuner built 657 training graphs with 828,423 labeled candidate pairs (15,433 positive) and 167 validation graphs with 186,418 labeled candidate pairs (3,195 positive). These exactly match the eligible pair counts used by the Phase 5 logistic-regression baseline. Test data was not used for tuning.

Simple words: In Step 5, I tried four small model settings, changing the embedding size and learning speed. I compared them using only the validation sessions and kept the one that ranked future conflicts best. The test session stayed untouched.

Observations:
- 64 hidden channels and a 0.001 learning rate gave the best validation average precision of the four tried settings.
- The result is a tuning choice, not a test result. Step 6 retrains using the selected settings, saves the model and only then generates test predictions.
- The grid is intentionally small to keep tuning practical; it does not claim to find the globally best GraphSAGE configuration.

## Phase 6, Step 6 — Save the Best Model, Configuration and Test Predictions

Run `python -m src.gnn.future_tuning --finalize` after Step 5's tuning command. It loads the validation-selected configuration, fits feature scaling on training graphs only, trains on training data, and keeps the epoch with the best validation average precision. The test split is scored only after model selection.

| Saved item | Location |
|---|---|
| Trained model weights, configuration, selected epoch, validation score and training-only scaler statistics | `models/future_graphsage.pt` |
| Readable model configuration and pair counts | `data/processed/graphsage_tuning/best_config.json` |
| One future-conflict probability per test pair, with snapshot time, aircraft IDs and target | `data/processed/graphsage_tuning/test_predictions.csv` |

The final model used 64 hidden channels, learning rate 0.001, 2 layers, dropout 0.2, mean aggregation, and the best validation epoch (5). Best validation AP in the latest final training run was 0.2577. The test output contains 105,856 probabilities for 1,577 positive pairs; all probabilities are finite and between 0 and 1, with no duplicate pair rows.

**Same-pair check:** the GraphSAGE test predictions have exactly the same 105,856 `(snapshot_time, icao24_a, icao24_b)` rows and labels as the Phase 5 six-feature logistic-regression test predictions: no missing or extra pairs. This also verifies Step 7's candidate-pair comparison requirement.

Simple words: In Step 6, I saved the trained GraphSAGE model, wrote down its settings, and made a probability for every test pair. I checked that it scored exactly the same pairs as the main logistic-regression baseline, so the comparison is fair.

Observations:
- Model selection used validation data only. Test labels were not used to select the settings or best epoch; they are included beside probabilities only for later evaluation.
- The model and generated predictions are Git-ignored. Rerun the tuning command followed by `python -m src.gnn.future_tuning --finalize` to regenerate them.
- The selected model achieved a final-run validation AP of 0.2577, while the tuning-grid run reported 0.2550 for the same settings. Each figure comes from a separate training run; neither number uses test data.

## Phase 6, Step 7 — Confirm Candidate-Pair Alignment

The finalization command now checks [`data/processed/graphsage_tuning/test_predictions.csv`](data/processed/graphsage_tuning/test_predictions.csv) against the Phase 5 six-feature baseline's [`logreg_test_predictions.csv`](data/processed/future_predictions/logreg_test_predictions.csv). It compares the full test candidate-pair identifiers (`snapshot_time`, the two aircraft IDs, treated as an unordered pair) and the `future_conflict` labels, and raises an error if either prediction file has duplicates or if any pair or label differs.

The check passed: both models score exactly **105,856 test candidate pairs**, with identical IDs and labels (1,577 positive pairs). There are no missing pairs, extra pairs or duplicate rows. Run `python -m src.gnn.future_tuning --finalize` to regenerate GraphSAGE predictions and repeat the check. It requires the Phase 5 baseline test-predictions file to exist first.

Simple words: In Step 7, I checked that GraphSAGE and logistic regression took the same test. Both scored the same aircraft pairs at the same times and used the same answers. That makes their comparison fair.

Observations:
- The check ignores pair order (A–B and B–A are treated as the same pair), but it does check the snapshot time and label.
- Pair alignment is now checked automatically every time the finalization command runs, not just by a one-time manual comparison.

## Phase 6 Deliverable — Reproducible GraphSAGE Future-Conflict Pipeline

**Reproduce the pipeline:**

1. Run `python -m src.gnn.future_tuning` to compare the four documented settings using training data for fitting and validation average precision for selection only. This saves the tuning table and selected settings under `data/processed/graphsage_tuning/`.
2. Ensure the Phase 5 baseline test predictions exist at `data/processed/future_predictions/logreg_test_predictions.csv`.
3. Run `python -m src.gnn.future_tuning --finalize` to fit the selected model, select its best epoch with validation data, save the model and configuration, produce test probabilities, and verify that the GraphSAGE predictions contain exactly the same candidate pairs and labels as the logistic-regression comparison.

**Pipeline files:** [`src/gnn/future_tuning.py`](src/gnn/future_tuning.py) loads the split data, trains and tunes GraphSAGE, and writes predictions. [`src/gnn/data.py`](src/gnn/data.py) builds sparse PyTorch Geometric snapshot graphs and aligns future labels to candidate edges. [`src/gnn/model.py`](src/gnn/model.py) creates contextualized aircraft embeddings and edge-classifier scores.

| Deliverable item | Saved location |
|---|---|
| Tuning results and selected configuration | `data/processed/graphsage_tuning/tuning_results.csv`, `best_config.json` |
| Trained model weights, model settings and training-only scaler values | `models/future_graphsage.pt` |
| Test future-conflict probabilities, labels, snapshot times and aircraft pair IDs | `data/processed/graphsage_tuning/test_predictions.csv` |
| Candidate-pair alignment check | Runs during finalization against `data/processed/future_predictions/logreg_test_predictions.csv` |

The current test prediction file contains **105,856 candidate pairs**, including **1,577 positive future-conflict labels**, and one finite probability between 0 and 1 for every pair. The pair IDs and labels match the logistic-regression test set exactly. Model and generated outputs are Git-ignored and can be regenerated with the commands above.

Simple words: The GraphSAGE pipeline can be run again from the saved project data. It first chooses settings using validation data, then trains and saves the selected model, and finally gives a future-conflict probability to every test candidate pair. It checks that logistic regression got the same test pairs.

Observations:
- The test set is used only after settings and the best training epoch have been selected with validation data.
- The final probability file contains aircraft IDs and snapshot times, so every score can be traced to its candidate pair.
- The same-pair check makes the GraphSAGE and logistic-regression comparison reproducible and fair.