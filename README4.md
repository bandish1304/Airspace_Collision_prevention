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