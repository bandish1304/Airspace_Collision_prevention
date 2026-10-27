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
