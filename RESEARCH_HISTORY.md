# Maritime Trajectory Prediction Research Record

**Project:** `maritim`
**Research cycle:** corrected `ct_dma` reference → modular deterministic/multimodal forecasting → route-intent diagnostics → vessel-history context
**Status:** **CLOSED**
**Final selected model:** **B3A — deterministic K=1 with strictly historical same-vessel route context**
**Primary final-test result:** **12.035 ± 0.385 → 10.297 ± 0.456 nmi at 6 h**, a **14.4% reduction** versus B0 across seeds 123–125.

This document is the single consolidated research record for the completed model-development cycle. It replaces the individual experiment reports created during development. Its purpose is to let a future researcher quickly understand the benchmark, what was tried, what failed, what worked, what should not be repeated, and what remains worth studying.

The repository was deliberately simplified after the final test. Failed experiment implementations were removed; their scientific conclusions are retained here. Run artifacts remain under `runs/` as evidence.

---

## 1. Executive summary

The research began by correcting and validating the `ct_dma` TrAISformer reference pipeline, then building a much smaller direct trajectory predictor. The direct deterministic Transformer (B0) became the strongest deployable baseline: it substantially outperformed the TrAISformer stochastic mean at longer horizons while being far smaller and faster.

A K=16 multimodal model produced excellent candidate coverage, but its central weakness was ranking: useful trajectories were often present but assigned poor probability. Controlled experiments then tested stable route classes, semantic reranking, deterministic guidance, candidate-aware scoring, explicit route-intent supervision, a generic route atlas, heading, and rate of turn. None produced a reliable deployable improvement over the deterministic baseline.

The decisive new signal came from **strictly historical same-vessel route behavior**. A non-neural same-MMSI diagnostic showed that a vessel's own earlier trajectories contained route-choice information not available from recent motion alone. B3A integrated a small, missingness-aware summary of that historical route into the deterministic model.

B3A passed inner-development screening, three-seed replication, inference-time context ablation, untouched original-validation confirmation, and frozen final test. On final test it reduced mean 6 h FDE from **12.035 to 10.297 nmi** across seeds 123–125. The gain was largest when historical support existed and on difficult route-change cases. Removing historical context at inference significantly degraded B3A again, confirming that the model actually uses the history feature.

The main remaining limitation is **unseen vessels**, where B3A is slightly worse than B0. This motivates a separate future study on vessel-independent route context. No post-test model adjustment was made.

---

## 2. Benchmark, data, and evaluation protocol

### 2.1 Corrected `ct_dma` benchmark

Canonical split sizes:

| Split | Trajectories |
|---|---:|
| Train | 2,587 |
| Validation | 369 |
| Test | 438 |

Temporal setup:

- observed history: **18 steps**;
- forecast horizon: **72 steps**;
- cadence: **300 s / 5 min**;
- observed duration: **90 min**;
- forecast duration: **6 h**;
- evaluation horizons: **1 h, 2 h, 3 h, 6 h**.

Canonical AIS channels available in the processed data:

`LAT, LON, SOG, COG, HEADING, ROT, TIMESTAMP, VESSEL_MMSI, TRAJECTORY_ID`

The processed source does **not** contain destination, navigation status, ship type, draught, vessel dimensions, or port/destination identifiers.

The corrected data audit verified zero source/parquet mismatches and a constant 300-second cadence.

### 2.2 Coordinate and metric corrections

The corrected implementation fixed:

- the Danish ROI and five-minute cadence;
- normalization/denormalization semantics;
- stable trajectory ordering;
- haversine distance in nautical miles;
- explicit validity masks;
- prediction/target/trajectory-ID alignment;
- sequence length and masking semantics;
- deterministic validation/checkpoint behavior;
- reproducible frozen evaluation.

For modular models, normalized movement constraints were replaced by a physical displacement interpretation using the geographic coordinate scale. B0/B3A use the corrected deterministic physical decoder.

### 2.3 Development split

After repeated early validation studies, architecture development moved to a fixed chronological split of the original training data:

| Partition | Trajectories | Role |
|---|---:|---|
| `inner_train` | 2,069 | earliest 79.98% |
| `inner_dev` | 518 | latest 20.02% |

Split hash:

`07c52f33a9a5ccd37fa4c9cfc4e69093ad322771d4c74005bba85cb4d33d9807`

The original validation and test sets were then preserved for confirmation and final testing.

### 2.4 Core metrics

Primary deployable metric: mean endpoint error (FDE) in nautical miles at 1, 2, 3, and 6 h.

Other metrics used when relevant:

- complete-trajectory ADE;
- pointwise `minFDE@K`;
- complete-oracle FDE;
- `minADE@K`;
- ranking regret;
- top-M coverage;
- trajectory energy score;
- probability entropy/effective modes;
- clipping fractions;
- signed endpoint-deviation angular error;
- five-bin intent accuracy, balanced accuracy, macro F1;
- recall for true `|endpoint deviation| > 45°`.

Paired trajectory bootstrap comparisons used 10,000 resamples with fixed seed 123 unless otherwise stated.

---

## 3. Corrected TrAISformer reference

The corrected reference work established the benchmark before proposed-model search. It used corrected `ct_dma` semantics, stochastic 16-rollout evaluation, mean-of-16 as the primary stochastic forecast metric, strict checkpoint reload, and deterministic repeatability checks.

Final corrected reference test result:

| Horizon | Mean-of-16 FDE | Pointwise best-of-16 diagnostic |
|---|---:|---:|
| 1 h | 1.957 nmi | 0.580 nmi |
| 2 h | 4.375 nmi | 1.175 nmi |
| 3 h | 7.347 nmi | 1.950 nmi |

The two repeated evaluations were bit-identical.

Reference evidence:

`runs/20260811T095837Z-c8ef666d-blurfix/`

The old Lightning/reference pipeline was later removed during repository cleanup. The corrected TrAISformer model was subsequently restored as a plain-PyTorch executable baseline behind the same shared forecasting runner used by B0/B3A; its accepted checkpoint names and numerical forward/loss semantics were preserved exactly.

---

## 4. Initial modular-model study

### 4.1 Direct one-shot forecasting

Early validation results:

| Model | 1 h | 2 h | 3 h | 6 h FDE |
|---|---:|---:|---:|---:|
| Dead reckoning | 2.667 | 7.152 | 12.764 | 31.552 |
| Constant velocity | 2.712 | 7.198 | 12.929 | 31.717 |
| TrAISformer, 16-rollout mean, seed 123 | 2.052 | 4.453 | 7.310 | 17.812 |
| Direct Transformer, 3-seed mean | **1.955** | **3.895** | **5.881** | **11.601** |
| TCN K=16 top-1, 3-seed mean | 2.118 | 4.627 | 7.226 | 14.983 |

The direct K=1 model became the operational baseline.

### 4.2 Controlled module screen

At 3 h, seed 123:

| Variant | Top-1 FDE | minFDE@K | Energy | Decision |
|---|---:|---:|---:|---|
| Transformer K=1 | **5.878** | 5.878 | 5.878 | keep |
| Transformer K=16 | 7.426 | 2.154 | 4.658 | TCN preferable |
| GRU K=16 | 7.863 | 2.222 | 4.890 | reject |
| TCN K=16 | 6.791 | **2.108** | **4.512** | keep as multimodal diagnostic |
| Local-only | 7.306 | 2.160 | 4.945 | reject |
| Retrieval only | 9.248 | 3.670 | 7.857 | reject |
| Retrieval + refinement | 7.411 | 3.312 | 5.119 | reject |
| Query + retrieval hybrid | 6.737 | 3.288 | 5.014 | reject |
| Soft probability targets | 7.083 | 2.036 | 4.523 | no material gain |
| Probability loss weight 4 | 7.660 | 2.382 | 5.142 | reject |

K=16 was the best probabilistic operating point in the mode-count sweep; K=32 regressed.

### 4.3 Central multimodal finding

The K=16 model produced useful route alternatives but ranked them badly. It often contained a strong candidate while selecting a very poor one.

This established the central multimodal problem:

> **Candidate generation was much better than candidate ranking.**

---

## 5. Corrective cleanup before further research

Before continuing model search, the modular implementation was corrected:

- causal dilated TCN with receptive field covering the 18-step context;
- shared physical coordinate-scale utility;
- isotropic physical max-speed decoder;
- simplified K=1 semantics;
- stronger dataset/config validation;
- corrected ADE eligibility;
- coherent complete-oracle FDE;
- ranking regret/top-M diagnostics;
- entropy/effective modes;
- whole-trajectory energy;
- clipping diagnostics;
- aligned trajectory/vessel IDs;
- paired bootstrap comparison tools.

Corrected direct Transformer, three-seed validation means:

| Horizon | FDE |
|---|---:|
| 1 h | 1.88 |
| 2 h | 4.01 |
| 3 h | 6.02 |
| 6 h | 11.75 |

Corrected K=16 TCN at 6 h:

| Metric | Value |
|---|---:|
| top-1 FDE | 14.77 |
| pointwise minFDE@16 | 3.52 |
| complete-oracle FDE | 4.41 |
| ranking regret | 11.25 |
| trajectory energy | 4.23 |
| effective modes | ~12.9 |

The ranking problem survived the cleanup and was therefore not an implementation artifact.

---

## 6. C1 — stable-route supervision

**Question:** Would stable semantic route identities make the probability head easier to train?

Method:

- K=16;
- K-means on complete training futures;
- descriptor `[E3, N3, E6, N6]`;
- one stable route class per query.

Seed-123 result:

| Metric | WTA | C1 |
|---|---:|---:|
| 6 h top-1 FDE | 14.00 | **12.49** |
| ranking regret | 10.57 | **7.80** |
| minFDE@16 | **3.42** | 4.69 |
| complete-oracle FDE | **4.35** | 5.52 |
| minADE@16 | **2.18** | 3.15 |
| trajectory energy | **4.18** | 5.29 |
| effective modes | 12.59 | **1.93** |
| route-class accuracy | — | 72.51% |
| trajectories with clipping | 25.20% | **96.48%** |

**Decision:** rejected.

**Lesson:** stable route identity is learnable and helps early ranking, but should not own trajectory generation under the tested exclusive-regression design. Candidate geometry and clipping deteriorated severely.

---

## 7. C1.5 — semantic reranking

No training. Original WTA candidates were assigned to stable route classes from their own predicted geometry; C1 class probabilities were transferred to represented candidates.

Key diagnostics:

- mean unique route classes among 16 WTA candidates: **3.56**;
- true route class represented: **99.66%**;
- C1 predicted route represented: **97.29%**.

At 6 h:

| Metric | WTA | C1.5 |
|---|---:|---:|
| top-1 FDE | 14.00 | 13.96 |
| ranking regret | 10.57 | 10.54 |
| energy | **4.18** | 4.49 |
| minFDE@16 | 3.42 | 3.42 |

Best candidate in true class: **3.49 nmi**.
Best candidate in C1 predicted class: **8.14 nmi**.
Actual reranked top-1: **13.96 nmi**.

**Decision:** rejected.

**Lesson:** coarse semantic intent is compatible with WTA candidates, but cannot solve within-route ranking. There are separate route-class and within-class ranking failures.

---

## 8. C1.6 — deterministic-guided ranking

No training. WTA candidates were ranked by mean full-path haversine distance to the deterministic forecast.

At 6 h:

| Model/ranker | FDE |
|---|---:|
| original WTA | 14.00 |
| deterministic K=1 | 11.55 |
| C1.6 nearest-to-direct | **11.55** |

C1.6 improved WTA by **2.45 nmi**, CI `[-4.12, -0.90]`, but was tied with the deterministic model itself.

Diagnostics:

- deterministic stable-route accuracy: **75.95%**;
- deterministic route represented in every WTA set: **100%**;
- best WTA candidate within deterministic route class: **7.16 nmi**.

**Decision:** useful diagnostic, not a final model.

**Lesson:** deterministic K=1 contains strong macro-intent information, but geometric proximity cannot identify the best detailed candidate within that intent.

---

## 9. C2A — frozen-generator candidate-aware ranker

The frozen candidate scorer saw:

- WTA history embedding;
- candidate geometry;
- candidate minus deterministic trajectory;
- candidate/deterministic path distance.

It did not receive query identity or route class.

At 6 h:

| Model | FDE |
|---|---:|
| original WTA | 13.999 |
| deterministic K=1 | 11.551 |
| C1.6 | 11.548 |
| C2A | **11.509** |

Versus deterministic K=1: `-0.042 nmi`, CI `[-0.790, 0.720]`.

Failure split:

| Deterministic intent | C2A FDE | Direct FDE | Global oracle |
|---|---:|---:|---:|
| correct | **6.936** | 7.101 | 2.627 |
| wrong | 25.944 | **25.600** | 5.941 |

Large endpoint deviations: C2A **33.510**, direct **31.262**.

**Decision:** rejected as final improvement.

**Lesson:** candidate awareness improves WTA ordering but does not recover incorrect macro intent.

---

## 10. C2A-ND — remove deterministic guidance

C2A-ND retained only frozen WTA history plus candidate geometry.

On chronological `inner_dev` at 6 h:

| Model | FDE |
|---|---:|
| deterministic K=1 | **10.602** |
| frozen WTA | 12.393 |
| C2A-ND | 15.334 |

Wrong deterministic intent:

- C2A-ND: **26.960**;
- WTA: **22.660**;
- direct: **24.436**.

Large endpoint deviation:

- C2A-ND: **28.474**;
- WTA: **28.111**;
- direct: **25.338**.

**Decision:** rejected; candidate-ranker refinement closed.

**Lesson:** deterministic guidance was not over-anchoring the scorer. Removing it made ranking worse. The bottleneck had moved from scorer architecture to route-choice information.

---

## 11. C3A — explicit signed route-intent prediction

A causal TCN predicted:

`[cos Δθ3, sin Δθ3, cos Δθ6, sin Δθ6]`

from the same 90-minute history.

Fair B0 `inner_train → inner_dev` comparison at 6 h:

| Metric | B0 | C3A |
|---|---:|---:|
| angular MAE | **10.39°** | 12.28° |
| five-class accuracy | **80.39%** | 72.64% |
| true >45° MAE | **33.87°** | 38.59° |
| true >45° recall | **49.21%** | 34.92% |

C3A carried some complementary information on B0 wrong-bin examples, but recovered the correct bin only 18.52% of the time.

**Decision:** rejected and closed.

**Lesson:** direct trajectory learning already produces a better usable intent representation than a separate history-only angular head. The next step required new information, not another head.

---

## 12. B0 fair control and context audit

B0 is the corrected deterministic Transformer K=1 trained only on `inner_train` and selected on `inner_dev`.

Seed-123 `inner_dev` FDE:

| Horizon | FDE |
|---|---:|
| 1 h | 1.645 |
| 2 h | 3.640 |
| 3 h | 5.590 |
| 6 h | 12.485 |

B0 contains 734,480 trainable parameters in this experiment configuration.

The context audit found no destination or vessel metadata that directly encoded voyage intent. Available unused information was:

- true heading;
- rate of turn (ROT);
- repeat-vessel history through MMSI.

---

## 13. C4A — coarse train-only route atlas

A non-neural route atlas used:

- 10 nmi spatial cells;
- eight incoming COG sectors;
- empirical 3 h/6 h downstream branches;
- modal five-bin branch plus within-branch circular mean;
- nearest populated cell in the same heading sector when an exact state was absent.

At 6 h:

| Metric | B0 | C4A |
|---|---:|---:|
| angular MAE | **10.39°** | 12.93° |
| five-class accuracy | **80.39%** | 72.64% |
| >45° recall | **49.21%** | 30.16% |
| true >45° MAE | **33.87°** | 38.79° |

**Decision:** rejected.

**Lesson:** a coarse cell/heading atlas is not rich enough for branch-level route decisions. This does not rule out a genuine directed route/corridor graph.

---

## 14. B1 — heading and rate-of-turn feature study

### Data audit

Heading:

- ordinary AIS degrees `0–359`;
- no `511` unavailable sentinel;
- encoded as `sin/cos`.

ROT:

- signed decoded physical AIS ROT;
- range `[-708.7, 708.7]`;
- no raw `-128` sentinel;
- rare saturation values;
- inner-train absolute p95: `14.25908333333328`.

Frozen ROT transform:

`tanh(ROT / 14.25908333333328)`

### Seed-123 screen

| Model | Added feature | 6 h FDE |
|---|---|---:|
| B0 | none | **12.485** |
| B1-H | heading | 12.675 |
| B1-R | ROT | 12.758 |
| B1-HR | heading + ROT | 13.486 |

B1-R was provisionally selected because it improved the predeclared B0 wrong-intent subgroup:

`30.92 → 27.67 nmi`, difference `-3.25`, CI `[-5.71, -0.97]`.

### Three-seed replication

Mean 6 h FDE:

| Model | Mean ± SD |
|---|---:|
| B0 | **12.719 ± 0.273** |
| B1-R | 13.041 ± 0.245 |

Each seed's own B0 wrong-bin cohort:

- seed 123: `-3.245 nmi`;
- seed 124: `+0.020 nmi`;
- seed 125: `-2.864 nmi`.

**Decision:** retain only as a narrow targeted signal; not selected as the final deterministic architecture.

**Lesson:** ROT contains some real maneuver-onset information, but the effect is seed-variable and does not improve overall long-horizon accuracy. Heading did not add useful overall value, and heading+ROT was harmful.

---

## 15. B2-R — ROT in K=16

B2-R added only the frozen signed ROT feature to the K=16 TCN.

Seed 123, inner split:

| 6 h metric | B2-0 | B2-R |
|---|---:|---:|
| top-1 FDE | 14.817 | 14.509 |
| minFDE@16 | **3.391** | 3.471 |
| complete-oracle FDE | **4.360** | 4.585 |
| ranking regret | 11.426 | 11.038 |
| energy | **4.198** | 4.228 |

Top-1 difference: `-0.308 nmi`, CI `[-1.592, +0.983]`.

Both difficult populations worsened:

- fixed B0 wrong-bin IDs: `27.615 → 29.111`;
- true >45°: `29.762 → 32.105`.

**Decision:** rejected. K=16 + ROT closed; C2-R not run.

**Lesson:** ROT's targeted deterministic signal did not translate into a useful multimodal improvement.

---

## 16. Strict same-MMSI historical-route diagnostic

### Question

Does a vessel's own earlier behavior contain route-choice information beyond a generic historical prior?

### Retrieval rule

For each query:

- same one of eight incoming COG sectors;
- nearest physical forecast origin;
- generic prior: any historical vessel;
- same-MMSI prior: identical MMSI;
- historical source's target endpoint timestamp strictly earlier than query forecast origin.

No future overlap was allowed.

### Coverage

On `inner_dev`:

- seen MMSI: **381 / 518 = 73.55%**;
- same-MMSI support at 6 h among valid seen cases: **66.0%**;
- median historical pool size: **2**;
- median match distance: **1.52 nmi**.

### Result

Matched supported 6 h cases (`N=200`):

| Prior | Angular MAE |
|---|---:|
| generic historical | 17.760° |
| same-MMSI | **13.092°** |

Difference: `-4.669°`, CI `[-8.963°, -0.490°]`.

Other diagnostics:

- balanced accuracy: `62.0% → 75.6%`;
- macro F1: `63.1% → 76.0%`;
- >45° recall: `47.2% → 72.2%`.

On 36 supported fixed B0 wrong-bin cases, same-MMSI history reduced angular MAE from **38.21° to 22.96°**.

**Decision:** informative; justify a learned missingness-aware vessel-history feature.

**Lesson:** the useful signal is historical behavior of the same vessel, not merely generic location/heading topology and not a raw MMSI embedding.

---

## 17. B3A — missingness-aware vessel-history context

### 17.1 Hypothesis

A vessel's strictly historical behavior in geographically and directionally similar states provides complementary long-horizon route-choice information beyond the current 90-minute trajectory.

### 17.2 Final design

B3A is B0 plus a small context branch.

Current trajectory:

- 18 observed steps;
- `LAT/LON/SOG/COG`;
- deterministic Transformer history encoder;
- K=1 physical displacement decoder.

Historical route eligibility:

1. same MMSI;
2. same one of eight incoming final-COG sectors;
3. source route's 6 h endpoint timestamp is strictly earlier than the query forecast origin.

Among eligible routes, choose the nearest physical forecast origin.

For training queries, only earlier training routes can be used.

For validation/test queries, historical context comes from the training split only. Validation/test trajectories never become history for other validation/test trajectories.

Context vector:

`[E3, N3, E6, N6, match_distance, log1p(pool_size), support]`

Context branch:

`7 → 32 → 32`

The branch is fused after the B0 history encoder with explicit support/missingness information.

Unsupported context is zero.

B3A does **not** use:

- ROT;
- heading;
- raw MMSI embedding;
- K=16;
- route classes;
- retrieval-distance cutoff;
- ensemble.

---

## 18. B3A inner-development result

Strict historical support:

| Population | Supported |
|---|---:|
| inner_train | 577 / 2069 = 27.89% |
| inner_dev | 236 / 518 = 45.56% |

Across seeds 123–125:

| Metric | B0 | B3A | Difference |
|---|---:|---:|---:|
| 1 h FDE | 1.665 ± 0.064 | 1.786 ± 0.150 | +0.121 |
| 2 h FDE | 3.606 ± 0.053 | 3.683 ± 0.153 | +0.077 |
| 3 h FDE | 5.563 ± 0.023 | 5.508 ± 0.124 | -0.055 |
| 6 h FDE | 12.719 ± 0.273 | **11.545 ± 0.592** | **-1.174 ± 0.713** |
| 6 h angular MAE | 10.231° | **9.117°** | **-1.114°** |
| balanced accuracy | 0.684 | **0.738** | +0.054 |
| macro F1 | 0.710 | **0.761** | +0.051 |
| >45° recall | 0.481 | **0.587** | +0.106 |

Supported 6 h improvement:

**-1.781 ± 0.731 nmi**

Hard supported groups:

- B0 wrong-bin + support: **-10.232 ± 3.120 nmi**;
- true >45° + support: **-9.144 ± 2.459 nmi**.

### Context ablation

Seed 123, supported inner-dev cases:

| Metric | Normal context | Context ablated |
|---|---:|---:|
| 6 h FDE | **11.793** | 13.861 |
| complete ADE | **5.351** | 5.872 |
| 6 h angular MAE | **9.540°** | 11.135° |

6 h normal-minus-ablated difference:

`-2.069 nmi`, CI `[-3.664, -0.660]`.

**Decision:** passed and replicated. B3A became the leading architecture candidate.

---

## 19. B3B — B3A + ROT complementarity

B3B changed only one thing: signed ROT was added to B3A.

Seed 123:

| Metric | B3A | B3B |
|---|---:|---:|
| 1 h FDE | **1.628** | 1.692 |
| 2 h FDE | 3.570 | **3.530** |
| 3 h FDE | **5.422** | 5.457 |
| 6 h FDE | **11.233** | 11.328 |
| complete ADE | **5.169** | 5.199 |
| 6 h angular MAE | **9.044°** | 9.455° |

B3B improved some intent-category metrics but did not improve forecasting, and unsupported performance worsened.

**Decision:** rejected after seed 123; seeds 124/125 not run.

**Lesson:** ROT does not add clear forecasting value beyond vessel-history context. Final model remains B3A without ROT.

---

## 20. Frozen B3A confirmation on untouched original validation

B0 and frozen B3A were trained on the complete original training split with seeds 123–125. Original validation was not incorporated into training.

Across seeds:

| Metric | B0 | B3A | B3A - B0 |
|---|---:|---:|---:|
| 1 h FDE | 1.881 ± 0.074 | 1.876 ± 0.060 | -0.005 |
| 2 h FDE | 4.014 ± 0.170 | **3.733 ± 0.139** | **-0.281** |
| 3 h FDE | 6.020 ± 0.195 | **5.596 ± 0.219** | **-0.424** |
| 6 h FDE | 11.748 ± 0.197 | **10.227 ± 0.362** | **-1.521 ± 0.543** |
| complete ADE difference | — | — | **-0.516 ± 0.316** |
| 6 h angular MAE | 9.475° | **8.169°** | **-1.305°** |
| balanced accuracy | 0.676 | **0.731** | +0.055 |
| macro F1 | 0.702 | **0.746** | +0.044 |
| >45° recall | 0.475 | **0.636** | +0.162 |

Seed-wise 6 h paired results:

| Seed | Difference | 95% CI |
|---|---:|---:|
| 123 | -0.906 | [-1.715, -0.128] |
| 124 | -1.725 | [-2.722, -0.718] |
| 125 | -1.934 | [-2.853, -1.025] |

Every seed improved significantly.

Validation historical support:

**168 / 369 = 45.53%**

Supported valid-at-6 h gain:

**-2.650 ± 0.655 nmi**

Hard supported gains:

- B0 wrong-bin + support: **-8.623 ± 2.694 nmi**;
- true >45° + support: **-13.979 ± 3.480 nmi**.

Seed-123 supported context ablation:

`9.035 → 11.780 nmi`

when history was removed.

Difference:

**-2.745 nmi**, CI `[-4.056, -1.536]`.

**Decision:** B3A confirmed; architecture frozen.

---

## 21. Final frozen test

No architecture, retrieval rule, threshold, checkpoint rule, training input, or feature was changed after validation confirmation.

Competing systems:

- B0;
- frozen B3A.

Existing full-original-training checkpoints for seeds 123–125 were used.

Test queries could retrieve only eligible original-training trajectories. Test trajectories never supplied history to other test queries.

### 21.1 Context availability

| Population | Count | Fraction |
|---|---:|---:|
| all test | 438 | 100% |
| historical support | 207 | 47.26% |
| seen MMSI, no support | 112 | 25.57% |
| unseen MMSI | 119 | 27.17% |

Supported history:

- pool size median: **2**;
- IQR: **1–4**;
- P90: **6**;
- maximum: **16**;
- match-distance median: **1.22 nmi**;
- IQR: **0.46–6.69 nmi**;
- P90: **38.96 nmi**;
- maximum: **126.76 nmi**.

No distance cutoff was added.

### 21.2 Primary final result

There are 348 test trajectories with valid 6 h targets.

Across seeds 123–125:

| Metric | B0 | B3A | Difference |
|---|---:|---:|---:|
| **6 h FDE** | 12.035 ± 0.385 | **10.297 ± 0.456** | **-1.739 ± 0.776 nmi** |

This is a **14.4% reduction** versus B0.

Seed-wise:

| Seed | B0 | B3A | Difference | Paired 95% CI |
|---|---:|---:|---:|---:|
| 123 | 11.740 | 10.822 | -0.918 | [-1.852, -0.00004] |
| 124 | 12.471 | 10.011 | -2.460 | [-3.586, -1.325] |
| 125 | 11.894 | 10.056 | -1.838 | [-2.831, -0.853] |

All three seeds are favorable and all three paired intervals are below zero.

### 21.3 Secondary test results

| Metric | B0 | B3A | Difference |
|---|---:|---:|---:|
| 1 h FDE | 1.782 ± 0.121 | **1.722 ± 0.049** | -0.060 |
| 2 h FDE | 3.770 ± 0.252 | **3.508 ± 0.147** | -0.262 |
| 3 h FDE | 5.828 ± 0.342 | **5.388 ± 0.258** | -0.440 |
| complete ADE difference | — | — | **-0.583 ± 0.438** |
| 6 h angular MAE | 9.119° | **7.878°** | -1.241° |
| balanced accuracy | 0.653 | **0.690** | +0.037 |
| macro F1 | 0.678 | **0.709** | +0.031 |
| >45° recall | 0.450 | **0.550** | +0.100 |

All frozen forecasting horizons improve on average.

### 21.4 Availability populations

| Population | B0 6 h FDE | B3A 6 h FDE | Difference |
|---|---:|---:|---:|
| supported | 13.145 ± 0.338 | **10.224 ± 0.696** | **-2.921 ± 1.023** |
| seen, no support | 12.207 ± 0.593 | **10.624 ± 0.450** | -1.583 ± 1.006 |
| unseen MMSI | **9.778 ± 0.412** | 10.160 ± 0.326 | **+0.382 ± 0.345** |

Primary limitation:

> **B3A is modestly worse on unseen vessels.**

### 21.5 Route-change cohorts

| Cohort | Count | Mean B3A - B0 6 h FDE |
|---|---:|---:|
| all true >45° | 40 | **-4.962 ± 1.577 nmi** |
| true >45° + support | 25 | **-7.696 ± 2.675 nmi** |
| true >45° unsupported | 15 | -0.407 ± 3.140 nmi |

### 21.6 Test context ablation

Seed 123, supported test cases:

| Metric | Normal context | Context ablated | Difference |
|---|---:|---:|---:|
| 6 h FDE | **11.023** | 13.709 | **-2.686**, CI `[-3.927, -1.438]` |
| complete ADE difference | — | — | **-0.593**, CI `[-0.890, -0.290]` |
| 6 h angular MAE difference | — | — | **-2.214°**, CI `[-3.600°, -0.906°]` |

The context mechanism therefore reproduces on the final holdout.

### 21.7 Clipping

| Diagnostic | B0 | B3A |
|---|---:|---:|
| trajectories with clipping | 7.08% ± 0.23% | 8.83% ± 1.65% |
| predicted points clipped | 2.56% ± 0.25% | 3.10% ± 0.53% |

Clipping rises modestly and should remain disclosed.

**Decision:** **B3A passes final test. Model development is closed.**

---

## 22. What worked

### 22.1 Direct one-shot K=1 forecasting

This was the first major improvement. Compared with the corrected TrAISformer-style stochastic forecast, the small direct model was substantially stronger at long horizons, much smaller, much faster, and easier to reproduce.

### 22.2 Physical-coordinate cleanup

The physical decoder and corrected evaluation semantics preserved the deterministic gain while making model behavior easier to interpret and audit.

### 22.3 Same-vessel historical behavior

This was the strongest genuinely new source of information discovered.

Evidence chain:

1. non-neural same-MMSI history beats a matched generic route prior;
2. B3A improves supported inner-dev cases in all seeds;
3. B3A improves difficult supported cases strongly;
4. inference-time context removal degrades the trained model;
5. the effect reproduces on untouched validation;
6. the effect reproduces on final test.

Final supported-test gain:

**-2.921 ± 1.023 nmi at 6 h**

---

## 23. What did not work

This section is deliberately explicit so a future researcher does not repeat closed experiments without new evidence.

### 23.1 Generic encoder search

GRU and larger K=16 Transformer variants did not solve the core problem. TCN was the best lightweight K=16 backbone, but ranking remained poor.

**Do not reopen GRU/Transformer/TCN search unless the information source changes.**

### 23.2 Naive retrieval/refinement/hybrid forecasting

Nearest-history retrieval and simple refinement/hybrid variants were not competitive.

B3A should not be confused with this earlier failure: B3A uses same-vessel strictly historical route context as an **auxiliary feature**, not retrieval as the forecast.

### 23.3 More modes

K=32 regressed relative to K=16.

### 23.4 Soft probability targets / probability-loss upweighting

No material ranking improvement.

### 23.5 Stable K-means route identities

C1 improved route classification and early ranking but harmed candidate geometry and caused extreme clipping.

**Do not bind generator queries to global stable route classes under the tested exclusive-regression design.**

### 23.6 Semantic reranking

C1.5 showed that stable classes are compatible with WTA candidates but too coarse to solve within-route ranking.

### 23.7 Deterministic nearest-path ranking

C1.6 is a strong diagnostic but only brings K=16 selection back to deterministic-model quality.

### 23.8 Candidate-aware ranking

C2A learns useful ordering but does not beat K=1 at 6 h. C2A-ND is much worse.

**Do not continue candidate-ranker depth/loss tuning with the same information.**

### 23.9 Explicit history-only route-intent prediction

C3A is worse than direct trajectory learning overall and on large deviations.

### 23.10 Coarse generic route atlas

C4A fails. This rejects a 10 nmi cell × heading-sector atlas, not route topology in general.

### 23.11 Heading

No useful overall addition.

### 23.12 ROT

ROT provides a narrow seed-variable signal on some deterministic failures, but:

- does not improve B0 overall across seeds;
- does not improve K=16;
- does not improve B3A when added as B3B.

ROT is therefore a diagnostic signal, not part of the final architecture.

---

## 24. Scientific interpretation

### 24.1 The multimodal oracle gap is real but not automatically exploitable

K=16 often contains excellent futures. The problem is not merely generating alternatives; recent history frequently does not provide enough usable information to know which alternative will occur.

### 24.2 The direct trajectory objective learns useful intent implicitly

C3A showed that explicit angle supervision did not outperform direct trajectory learning. The trajectory objective itself acts as useful structured supervision.

### 24.3 Some route-choice information is vessel-specific

B3A supports the hypothesis that a vessel's previous behavior around similar geographic/approach states can encode operational or route preference absent from the immediate 90-minute trajectory.

This should be described as **historical vessel-route context**, not simply “MMSI personalization.”

B3A does not use a learned MMSI embedding.

### 24.4 B3A's mechanism is support-dependent

The strongest gains appear where usable historical context exists and in supported difficult route-change cases.

Support should therefore always be reported.

### 24.5 Unseen vessels remain the cleanest unsolved problem

B3A is slightly worse on unseen MMSIs in final test. This is the most defensible target for the next independent research project.

---

## 25. Final canonical models

### 25.1 B0

B0 is the canonical deterministic baseline.

High-level design:

- 18-step `LAT/LON/SOG/COG` history;
- continuous global/local representation;
- Transformer history encoder;
- deterministic K=1 full-horizon physical decoder;
- masked physical trajectory loss;
- no vessel-history context.

Canonical config:

`configs/model/vessel_direct.yaml`

### 25.2 B3A

B3A is the final selected architecture.

Canonical config:

`configs/model/vessel_direct_b3a.yaml`

Core implementation:

- `models/direct/model.py`
- `models/direct/vessel_history.py`
- `forecasting/adapters.py`
- `forecasting/runner.py`
- `scripts/train.py`
- `scripts/evaluate.py`

B3A additions over B0:

- strictly historical same-MMSI route lookup;
- same one-of-eight final-COG sectors;
- nearest physical forecast origin;
- strict source 6 h endpoint timestamp `<` query origin;
- seven-value missingness-aware historical context;
- `7 → 32 → 32` context encoder fused after history encoding.

### 25.3 Executable corrected reference baseline

The corrected TrAISformer remains a paper baseline rather than a selected
proposed model. Its canonical executable implementation and config are:

- `models/baselines/traisformer.py`
- `configs/model/traisformer.yaml`

It uses the same `forecasting/runner.py`, evaluator, artifact layout, and
comparison tooling as B0/B3A.

---

## 26. Final repository state after cleanup

The repository was intentionally simplified after final test.

Removed:

- closed C1–C4 implementations;
- B1/B2/B3B experiment implementations;
- K=16/WTA research machinery;
- stable-route and candidate-ranker code;
- route-intent and route-atlas code;
- retrieval/refinement/hybrid model branches;
- GRU/TCN experimental configs;
- old TrAISformer/reference training/evaluation pipeline;
- one-off research scripts and experiment-only tests;
- redundant protocol/reproduction/TODO documentation.

Retained:

- corrected executable TrAISformer baseline and config;
- B0 config;
- B3A config;
- deterministic model path;
- vessel-history implementation;
- physical coordinate utilities;
- shared dataset/target preprocessing;
- corrected evaluation metrics;
- shared train/evaluate/compare scripts;
- raw Danish preprocessing/config;
- run/model-state artifacts as evidence.

Post-cleanup verification:

- retained and shared-framework test suite: **33 passed**;
- canonical-source Ruff: passed;
- Python compilation: passed;
- `git diff --check`: passed;
- B0 CPU smoke train/evaluate: passed;
- B3A CPU smoke train/evaluate: passed;
- frozen B0/B3A strict state-dict load: passed;
- accepted TrAISformer strict state-dict load: all 147 keys matched;
- old/new TrAISformer forward logits and training loss: exactly identical;
- full 16-rollout TrAISformer validation replay: 850,175/850,176 predicted coordinates exactly identical; one stochastic coordinate differed by one latitude bin, while 1/2/3/6 h mean FDE differed by less than `1.1e-7` nmi and the maximum 72-step curve difference was `0.000111` nmi;
- reconstructed B3A context: array-identical to frozen artifact;
- frozen validation prediction maximum delta:
  - B0: `4.77e-07`;
  - B3A: `2.38e-07`;
- probabilities exactly identical;
- test was not reloaded during cleanup.

The deliberately retained generic multimodal evaluation utilities are low-risk infrastructure, not active B3A model complexity.

---

## 27. Known limitations

### 27.1 Historical support is incomplete

Final test support:

**47.26%**

More than half of test trajectories therefore lack usable same-vessel historical context under the strict rule.

### 27.2 Unseen vessels

Final test unseen-vessel result:

`B3A - B0 = +0.382 ± 0.345 nmi`

This is the main final-model limitation.

### 27.3 Sparse history pools

Median supported pool size is **2** routes. The current method intentionally uses one nearest route.

More sophisticated multiple-history aggregation was not justified before final test and must be treated as future work.

### 27.4 Distant historical matches

Test match-distance P90 is **38.96 nmi**, maximum **126.76 nmi**.

Post-hoc Spearman correlations between match distance and B3A-minus-B0 error were near zero across seeds. No evidence justified adding a post-test distance cutoff.

### 27.5 Clipping

B3A modestly increases clipping relative to B0. This should remain disclosed.

### 27.6 Missing metadata

The current dataset lacks destination, ship type, draught, navigation status, port ID, and dimensions. These potentially useful sources of route-choice context could not be tested.

---

## 28. Future research directions

These are **new studies**, not unfinished tuning of B3A.

### 28.1 Vessel-independent route context for unseen/unsupported vessels

Highest priority.

Goal:

> provide route-choice context when same-vessel history does not exist.

Promising sources:

- directed shipping corridors;
- traffic-separation schemes;
- port/harbor topology;
- learned directed route graph from training AIS;
- static navigational constraints.

A future study should focus especially on:

- unseen MMSIs;
- seen MMSIs without valid history;
- unsupported true-large-deviation cases.

Do not simply repeat C4A with slightly different grid sizes. The representation should be genuinely route-structured.

### 28.2 Destination / port information

If future data includes destination or intended port, this is likely a high-value route-choice signal.

It was impossible to study with the current data.

### 28.3 Multiple historical voyages

If richer history becomes available, compare:

- current single nearest route;
- multiple same-vessel historical routes;
- empirical vessel-route distributions.

This should not become a raw MMSI embedding study.

### 28.4 Better missing-support neutrality

B3A is slightly worse on unseen vessels. A future architecture could explicitly preserve B0 behavior when context is unavailable, but this must be studied under a new development protocol rather than tuned on the completed final test.

### 28.5 Multimodality only after new information exists

The K=16 oracle remains scientifically interesting, but ranking should not be reopened using the same recent history alone.

A multimodal revisit is justified only if a new context source demonstrably improves route-choice information.

---

## 29. Directions that should remain closed unless genuinely new evidence appears

Do **not** reopen these merely as architecture tuning:

- GRU vs Transformer vs TCN search on the same inputs;
- K=32 or more modes;
- probability-loss weight tuning;
- soft WTA targets as the main ranking fix;
- stable K-means route classes bound directly to generator queries;
- semantic reranking with old WTA probabilities;
- deterministic nearest-path candidate selection as a final model;
- deeper candidate-aware scorer using the same history only;
- candidate attention without new route information;
- another history-only intent head;
- coarse cell × heading route atlas;
- heading as a simple extra feature;
- ROT-only final model;
- ROT in K=16;
- ROT added to B3A;
- hand-built ROT switches;
- post-test retrieval-distance cutoff.

These were already tested or directly ruled out by gated predecessor results.

---

## 30. Key run artifacts

The exact code for failed experiments has been removed, but run evidence remains.

### Corrected reference

- `runs/20260811T095837Z-c8ef666d-blurfix/`

### Initial modular research

- `runs/20260812-modular-validation-research/`

### Stable routes

- `runs/20260820-c1-stable-route-tcn-k16-seed123/`

### C1.5 semantic reranking

- `runs/20260820-c15-semantic-rerank-seed123/`

### C1.6 deterministic ranking

- `runs/20260820-c16-deterministic-rank-seed123/`

### C2A

- `runs/20260820-c2a-candidate-ranker-seed123/`

### C2A-ND

- `runs/20260820-c2a-nd-inner-seed123-r2/`

### C3A

- `runs/20260820-c3a-signed-route-intent-inner-seed123/`

### B0 / C4A

- `runs/20260820-b0-inner-direct-seed123/`
- `runs/20260820-c4a-route-atlas-inner/`

### B1 / B2 / same-MMSI

- `runs/20260821-b1-r-seed-replication.json`
- `runs/20260821-b2-rot-comparison.json`
- `runs/20260821-same-mmsi-route-diagnostic.json`

### B3A inner-development

- `runs/20260821-b3a-replication-seeds123-125.json`
- `runs/20260821-b3a-context-ablation-seed123.json`

### B3A frozen validation

- `runs/20260821-b3a-full-validation-replication.json`
- `runs/20260821-b3a-full-validation-context-ablation-seed123.json`

### Final frozen test

- `runs/20260821-frozen-b3a-test-context-availability.json`
- `runs/20260821-frozen-b3a-test-vessel-history-context.npz`
- `runs/20260821-frozen-b3a-final-test-replication-r2.json`
- `runs/20260821-frozen-b3a-test-context-ablation-seed123.json`
- `runs/20260821-frozen-b3a-test-seed{123,124,125}-r2.json`

Future researchers should treat these artifacts as historical evidence. The executable canonical systems are the corrected TrAISformer baseline, B0, and the selected B3A model; only B0/B3A are proposed deterministic research models from this cycle.

---

## 31. Final conclusion

The central result of this research cycle is not that a larger or more complex neural model improved maritime trajectory prediction.

The opposite happened.

A small deterministic direct model already provided a strong baseline. Multimodal models generated valuable alternatives but could not reliably rank them from recent motion alone. Stable route classes, semantic reranking, candidate-aware scoring, explicit intent supervision, coarse route topology, heading, and rate of turn did not solve the long-horizon route-choice problem.

The successful improvement came from adding **new information**:

> a strictly historical, missingness-aware summary of how the same vessel previously moved through a geographically and directionally related state.

That information reduced final-test 6 h FDE from:

**12.035 ± 0.385 nmi → 10.297 ± 0.456 nmi**

across seeds 123–125, while producing even larger gains on supported difficult route-change cases.

The final model is therefore:

> **B3A: deterministic K=1 trajectory prediction with strictly historical same-vessel route context.**

The research cycle is closed. Future work should focus on **vessel-independent route context for unseen and unsupported vessels**, not on reopening the closed scorer/intent/multimodal architecture searches without genuinely new information.
