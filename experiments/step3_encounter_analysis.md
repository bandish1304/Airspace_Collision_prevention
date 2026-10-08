# Are the positive examples distinct situations?

Answers the question raised in the 2026-09-24 progress feedback:

> whether the roughly 55 positive aircraft-pair examples represent 55 genuinely
> different risk situations, or whether many come from the same flights or event
> observed repeatedly across nearby timestamps.

**Short answer: they are heavily repeated. A ten-fold inflation at the level the
question asks about, and worse once spatial overlap is also accounted for.**

---

## Why the dataset changed first

The question cannot be answered on the Week 1 data. That dataset is a **single
snapshot** — one timestamp, 302 aircraft — so there are no "nearby timestamps"
for an event to repeat across. The 55 positives reported previously are 55
observations of one instant, and the only possible duplication in them is
spatial.

A continuous window was therefore collected before any analysis:

| | |
|---|---|
| Source | OpenSky `/states/all`, one-hour history window |
| Captured | 2026-09-25, 14:38:50 – 15:36:50 PDT |
| Frames | 349 at 10-second spacing, no gaps |
| Aircraft | 752 airborne, 117,298 state rows |
| Pair-observations | 4,158,918 |

Labelling is unchanged from Week 2: lateral ≤ 5 nm **and** vertical ≤ 1,000 ft.

The multi-month archive needed for the sequence models is a separate matter and
still depends on OpenSky Trino access, which remains pending.

---

## Headline numbers

| | |
|---|---|
| Positive pair-observations | 24,016 (0.577%) |
| Distinct encounters (same aircraft pair, time-contiguous) | **2,394** |
| **Inflation** | **10.0×** |
| Median observations per encounter | 7 |
| Median encounter duration | 60 s |
| Longest encounter | 20.5 min → 112 positives |
| Encounters lasting a single frame | 195 (8.1%) |
| Encounters exceeding 5 minutes | 120 |

An encounter ends when the pair separates for more than 60 seconds; returning
after that counts as a new encounter.

## The imbalance depends entirely on the unit of counting

| Unit | Positive rate |
|---|---|
| Pair-observations | 0.577% |
| Distinct aircraft pairs (2,078 of 78,400) | **2.65%** |

The same data, 4.6× apart. The pair-observation figure — the one previously
reported as 0.66% — is the most pessimistic framing available, because it counts
a single twenty-minute encounter as dozens of independent rare events. At the
event level the positive class is meaningfully less rare than earlier reports
suggested.

---

## Figures

### `step3_encounter_anatomy.png`

Left: how many positive rows each encounter generates. Right: how long
encounters last.

**Interpretation.** The modal encounter produces two to ten rows and the tail
runs past forty, which is the mechanism behind the 10× inflation: nothing about
the labelling rule is wrong, but a situation that persists is counted once per
frame. Most encounters are short — over half resolve inside a minute — yet 120
exceed five minutes and those long ones contribute disproportionately to the
positive count.

### `step3_single_encounter.png`

Lateral and vertical separation for aircraft `aa9bfd` and `ab2056` over the full
life of their encounter.

**Interpretation.** These two aircraft hold between 0.5 and 2.7 nm laterally and
under 800 ft vertically for twenty-one unbroken minutes, tracking each other's
manoeuvres throughout. Both thresholds are breached in all 112 frames, making
this the single largest contributor to the positive class in the window. It is
almost certainly formation or paired traffic rather than a conflict. The figure
is the clearest available illustration of two separate problems: that one
situation yields many near-identical training rows, and that the current rule
cannot distinguish deliberate close formation from loss of separation.

### `step3_positive_concentration.png`

Connected components of the positive-pair graph, where nodes are
`(aircraft, frame)` and edges join the two aircraft of a positive pair and the
same aircraft across consecutive frames.

**Interpretation.** The 24,016 positives form 508 components, but one component
alone contains 11,882 of them — 49.5%, spanning 217 aircraft and the entire
58-minute window. This is a transitive-chaining effect rather than a single
encounter: A is near B, B near C, C near D, and the chain propagates across the
dense Los Angeles basin. The practical consequence is that the positive class
has very little independent structure. The top five components hold 67.5% and
the top twenty-five hold 79.8%.

---

## Consequence for train/test splitting

The feedback correctly linked this question to how the data should be split.
Three points follow from the numbers above.

1. **A random row-level split leaks.** Frames from the same encounter land in
   both halves. The model is scored on aircraft pairs it has already seen ten
   seconds earlier at nearly identical coordinates. The Week 3 results were
   produced under exactly this split and should be regarded as optimistic until
   re-run.

2. **A grouped split is not available.** Keeping components intact places 49.5%
   of the positive class wholly in train or wholly in test. No group-based
   k-fold can balance that.

3. **A temporal split is the workable option** — train on the earlier part of
   the window, test on the later. Proximity chains do not cross a time boundary
   the way they cross a group boundary, and a temporal split also matches the
   intended use of the model, which is prediction forward in time.

---

## Two further observations

**Roughly half the positive class is not converging.** 51.5% of positive
pair-observations have a positive closing speed; the remainder are aircraft
moving apart, which cannot collide. The rule considers only current separation
and ignores the closing speed and time-to-CPA that the pipeline already
computes.

**The positives are almost entirely low-altitude.** 99.4% occur below 10,000 ft,
concentrated in terminal areas. Aircraft near airports fly close together as a
matter of normal procedure, so a large share of the positive class reflects
routine operations rather than risk.

Both point at the labelling rule rather than the data, and both are open
questions for the next discussion.

---

## Caveats

- One 58-minute window on a Friday afternoon. Time-of-day and day-of-week
  effects are not covered; scheduled sampling across several days is in progress.
- The 60-second encounter-gap threshold is a judgement call, but the result does
  not hinge on it. Sweeping the threshold across a 60× range moves the count only
  between 2,814 (10 s, where any missing frame splits an encounter) and 2,151
  (600 s), against a hard lower bound of 2,078 distinct aircraft pairs when time
  is ignored entirely. Inflation stays between 8.5× and 11.2× throughout.

  | Gap threshold | Encounters | Inflation |
  |---|---:|---:|
  | 10 s | 2,814 | 8.5× |
  | 30 s | 2,512 | 9.6× |
  | 60 s | 2,394 | 10.0× |
  | 120 s | 2,282 | 10.5× |
  | 600 s | 2,151 | 11.2× |
- Component counts are sensitive to the chaining effect described above and
  should not be read as a count of physical encounters.

## Reproducing

| Artifact | Path |
|---|---|
| Raw frames | `data/raw/sessions/states_socal_backfill_1790375809.csv` |
| Pair features | `data/processed/states_socal_backfill_1790375809_pairs.csv` |
| Encounters | `experiments/eda/step3_encounters.csv` |
| Situation assignment | `experiments/eda/step3_situation_assignment.csv` |

Collection: `python -m src.data.backfill_opensky --window-minutes 58 --step-seconds 10`
Pair features: `python -m src.features.build_pipeline <frames csv>`

`build_pair_features` was rewritten for this analysis to pair within each
snapshot rather than across a whole file, and vectorised over NumPy arrays.
Output on the Week 1 snapshot is identical to the previous implementation
(8,309 rows, same order, differences at 1e-11 floating-point level) and runs
roughly 165× faster.
