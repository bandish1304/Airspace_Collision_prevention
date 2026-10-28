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
