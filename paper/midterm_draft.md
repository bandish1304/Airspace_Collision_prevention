# Comparing Graph and Temporal Models for Aircraft-Pair Collision Risk from ADS-B Data

Midterm rough draft. Author, affiliation and date to be added.

## Abstract

This project asks whether a machine-learning model can flag risky aircraft pairs using ADS-B data collected over Southern California. Each pair of aircraft within 50 nautical miles gets a set of kinematic features, and a pair counts as risky when the aircraft are within 5 nautical miles laterally and 1,000 feet vertically. That rule makes the positive class rare, at about 0.6% of pair-observations. We are comparing a GraphSAGE graph neural network (GNN) with a Temporal Fusion Transformer (TFT) that reads 30-observation windows of a pair's history. So far the GNN baseline is finished. The TFT has a working pipeline and a first hyperparameter search scored on validation data only, but it has not been trained to convergence or tested on the held-out session. On that session GraphSAGE did not beat a plain logistic regression. Both models mostly recover the labelling rule, since two of their input features define it.

## 1. Introduction

Collision risk is a property of two aircraft together, and it builds up over time. Distance, altitude, heading, speed and the projected point of closest approach all matter at once. That points to two kinds of model. A GNN treats aircraft as nodes and nearby pairs as edges. A TFT treats a pair's recent history as a sequence.

We started with three questions. Which imbalance-handling strategy works best for a rare, safety-critical label? How does a graph model compare with a simple baseline when evaluation respects time order? And can a TFT be adapted to predict a binary risk label on the same data? This draft answers the first two and reports early work on the third. The direct GNN-versus-TFT comparison comes later in the semester.

## 2. Related Work

To be written. The papers I plan to start from are the Temporal Fusion Transformer (Lim et al., 2021), GraphSAGE (Hamilton et al., 2017), SMOTE (Chawla et al., 2002) and Optuna (Akiba et al., 2019), plus some of the aviation conflict-detection literature. I still need to read these properly and check the citations.

## 3. Data and Labels

The ADS-B state vectors come from the OpenSky Network. We have one 58-minute backfill of 349 frames and four 30-minute polling sessions recorded on 2026-10-07. Aircraft on the ground and stale position reports are removed. Pairs are built inside each frame, so two aircraft are never paired across different times. For each pair we compute horizontal distance, closing speed, bearing difference, vertical separation and time to closest approach.

A pair-observation is labelled positive when lateral separation is at most 5 nautical miles and vertical separation is at most 1,000 feet. This is a proximity rule. It does not identify real collisions, and the final label definition is still waiting on the professor's input.

Exploring the data showed a few things that shape the rest of the work. The positives are not independent. In the backfill, the 24,016 positive observations group into 2,394 encounters (observations of the same pair no more than 60 seconds apart), and the median encounter lasts only 60 seconds. One connected group of pairs holds about half of all positives, so a random row-level split could put related rows on both sides of the train/test boundary. How rare the positives look also depends on what we count: 0.58% of pair-observations, 2.66% of aircraft pairs, and 79% of aircraft. Risky pairs turned out to be mostly low, slow aircraft that happen to be close. Closing speed and bearing difference barely separate them from comparable safe traffic. When we removed the two features that define the label, AUROC fell from about 0.999 to about 0.75 and average precision from about 0.90 to 0.018.

## 4. Methods

### 4.1 Evaluation setup

We keep whole sessions together and order them in time. The backfill and the 6:39 AM and 10:18 AM sessions are used for training, the 2:30 PM session for validation and the 7:39 PM session for testing. The test session stays untouched until the final comparison. Decision thresholds are picked on validation by minimising a normalised cost in which a missed positive counts as five false alarms. A cost of 0 is perfect, and 1 is no better than never predicting risk.

### 4.2 Choosing an imbalance strategy

In Week 3 we compared a plain baseline, SMOTE, class weighting and focal loss on the backfill, using both a temporal and a random split. On the temporal split SMOTE had the lowest normalised cost (0.203, against 0.275 for the baseline) and missed none of the validation positives. Class weighting came close at 0.238. Focal loss predicted no positives at all. That comes from how it was configured (weights of 0.25 and 0.75), so it says little about focal loss itself. We carried SMOTE forward.

### 4.3 GNN baseline

The GraphSAGE model classifies edges, meaning aircraft pairs, in a graph of nearby aircraft. Feature scaling is fit on training graphs only, and we keep the checkpoint with the best validation average precision over five epochs. A logistic regression trained on the same six pair features and the same frames serves as the simple reference. In a follow-up run we applied SMOTE to the learned pair representations, not to physical nodes or edges. Validation and test data are never resampled.

### 4.4 Temporal Fusion Transformer

Each TFT example is 30 consecutive observations of one aircraft pair. The target is whether that pair meets the risk rule at least once in the next six observations. Windows never cross a change of pair, a gap longer than 30 seconds, a session boundary or a split boundary.

The model is PyTorch Forecasting's `TemporalFusionTransformer` with its output replaced by a single logit. We use a custom binary cross-entropy loss and apply a sigmoid at prediction time to get a risk probability. The target is not rescaled.

SMOTE is applied to the training windows only. We flatten each window, interpolate, and reshape it back. We allow at most 10 negatives per positive and aim for a positive-to-negative ratio of 0.5. Validation and test windows are left as they are.

For tuning we ran an Optuna study (TPE sampler, seed 42) that maximised validation PR-AUC. It searched over attention heads (1, 2 or 4), hidden size (8, 16, 32 or 64), dropout (0.05 to 0.4) and learning rate (1e-4 to 1e-2, log scale). There were 12 trials of 3 epochs each on CPU. Each trial used 8,000 SMOTE training windows (4,000 positive) and 20,000 real validation windows, of which 213 were positive. The test split was never loaded.

## 5. Results

### 5.1 GNN baseline on the held-out test session

The test session has 4,460 positives among 763,009 pair-observations.

| Model | Threshold (from validation) | Average precision | AUROC | Precision | Recall | Normalised cost |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.132 | 0.933 | 0.9995 | 0.668 | 0.954 | 0.140 |
| GraphSAGE (class-weighted) | 0.995 | 0.895 | 0.9993 | 0.624 | 0.946 | 0.168 |
| GraphSAGE (latent SMOTE) | 0.923 | 0.837 | 0.9987 | 0.545 | 0.943 | 0.215 |

GraphSAGE did not beat the simpler model on this session. Latent SMOTE pushed recall to 0.997 at a fixed 0.5 threshold, but it also produced 17,229 false positives, which gave a normalised cost of 0.776. All three models score very high because the label is a function of two of their inputs. The numbers show how well they reproduce the rule. They do not show that they can predict collisions.

### 5.2 TFT data pipeline

| Split | Windows | Positive windows |
|---|---:|---:|
| Train (before SMOTE) | 2,308,133 | 23,406 |
| Train (after SMOTE) | 351,090 | 117,030 (93,624 synthetic) |
| Validation | 403,376 | 4,513 |
| Test | 234,282 | 2,442 |

We ran a one-epoch smoke test on a small sample of 512 training windows and 256 validation windows. It finished end to end, with a training loss of 0.659 and a validation loss of 0.655. That only confirms the pipeline runs. It is not a measure of how well the model predicts.

### 5.3 TFT hyperparameter search

These scores come from validation data only.

| Rank | Attention heads | Hidden size | Dropout | Learning rate | Validation PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 64 | 0.24 | 0.0029 | 0.842 |
| 2 | 2 | 64 | 0.14 | 0.0045 | 0.819 |
| 3 | 2 | 64 | 0.39 | 0.0041 | 0.810 |

Learning rates between 0.003 and 0.005 worked best. The lowest rates (1e-4 to 3e-4) did worst, with PR-AUC between 0.31 and 0.58, probably because 3 epochs is too short for them to learn much. All three of the top trials used hidden size 64 with 2 attention heads. The best setting goes forward to Week 9. The full trial table is in `experiments/eda/week8_optuna_trials.csv` and the best settings are in `experiments/eda/week8_optuna_best.json`.

## 6. Discussion and Limitations

The biggest caveat is the label. Because it is a proximity rule built from two input features, high scores mostly show that a model can recover that rule. A more useful label would involve predicted separation at the closest point of approach, or a confirmed event definition.

The data is also thin. Everything comes from one date, and validation and test are one session each, so the results may not carry over to other days or seasons.

The tuning results need the same caution. The validation sample has only 213 positives, and the top trials are close enough that their differences are small next to that noise. The search used 3-epoch trials on 8,000 of the 351,090 training windows, which favours higher learning rates, so the best setting may change once training runs longer. A validation PR-AUC of 0.84 is not a test result.

Two further points are still open. SMOTE on flattened windows can create trajectories that are not physically realistic, and we have not yet measured whether that hurts the TFT. And the TFT has not been evaluated on the held-out test session at all.

## 7. Plan for the Remaining Weeks

Next I will train the chosen TFT configuration to convergence on the full SMOTE training set, with early stopping and learning-rate scheduling, and log it to MLflow (Week 9). Then both tuned models are evaluated once each on the same test session (Week 10). After that I want to compare against the logistic regression, try alternative label definitions, and look at the TFT's variable-selection and attention weights for interpretability. The last step is finishing the paper.

## References (still to verify)

- Lim, B., Arık, S. Ö., Loeff, N., Pfister, T. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4).
- Hamilton, W., Ying, Z., Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS*.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *KDD*.
