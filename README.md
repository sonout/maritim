# Maritime trajectory prediction

This repository provides one comparison framework for three executable
maritime forecasting systems:

- **TrAISformer** (`configs/model/traisformer.yaml`): corrected autoregressive
  categorical baseline with 16 stochastic rollouts;
- **B0** (`configs/model/vessel_direct.yaml`): deterministic direct Transformer;
- **B3A** (`configs/model/vessel_direct_b3a.yaml`): B0 plus strictly historical
  same-vessel route context.

They share split loading, target extraction, seed handling, training and
checkpoint control, validation/test alignment, geographic metrics, prediction
artifacts, and command-line entry points. Model adapters contain only genuine
algorithm differences: TrAISformer uses categorical teacher forcing and
autoregressive sampling; B0/B3A use masked physical trajectory regression and
direct full-horizon decoding.

## 1. Environment

Run commands from the repository root. The minimal runtime environment is
specified in [`environment.yml`](environment.yml). It contains only Python,
PyTorch, NumPy, pandas, PyArrow, and PyYAML. The verified environment used
Python 3.14, PyTorch 2.11, NumPy 1.26, pandas 2.3, PyArrow 24, and PyYAML 6.
Python 3.9 is not supported. GPU training requires a PyTorch build matching
the local CUDA installation.

Create it with Conda:

```bash
conda env create -f environment.yml
conda activate maritim
```

Verify that the active interpreter is the new environment before running an
experiment:

```bash
which python
python --version
python -c "import torch, numpy, pandas, pyarrow, yaml; print(torch.__version__); print(torch.cuda.is_available())"
```

The optional `pytest` and `ruff` tools are not needed to train or evaluate.
Install `matplotlib` separately only when using `scripts/plot_routes.py`:

```bash
python -m pip install matplotlib
```

For the optional repository verification suite, install its two extra tools:

```bash
python -m pip install pytest ruff
```

Then run:

```bash
python -m pytest -q
ruff check forecasting evaluation models/direct models/baselines scripts tests
python -m compileall -q forecasting evaluation models scripts tests
git diff --check
```

## 2. Data

The canonical experiments use the CT-DMA profile in
`configs/dataset/ct_dma.yaml`. Set `preprocessed_folder` to a directory
containing:

```text
train.parquet
valid.parquet
test.parquet
```

The files must contain the canonical per-trajectory array columns
`LAT/LON/SOG/COG/TIMESTAMP/VESSEL_MMSI/TRAJECTORY_ID`. The loader also accepts
the recorded legacy CT-DMA column mapping and validates alignment, five-minute
cadence, finite values, vessel consistency, and normalized coordinates.

The Danish data profile and preprocessing are retained under `preprocessing/`.
They are not silently treated as CT-DMA: a future dataset adapter must define
its split artifacts and normalization explicitly before using the common
forecast runner.

## 3. Quick preflight

These commands run one epoch on small train/validation slices and do not load
the test split. They verify mechanics only and are not research results:

```bash
python scripts/train.py --config traisformer \
  --run-dir runs/smoke-traisformer --device cuda:0 --smoke

python scripts/train.py --config vessel_direct \
  --run-dir runs/smoke-b0 --device cpu --smoke

python scripts/train.py --config vessel_direct_b3a \
  --run-dir runs/smoke-b3a --device cpu --smoke
```

Run directories are never reused. Choose new paths or remove disposable smoke
runs before repeating them.

## 4. Train all canonical systems

One command trains any model; only the config changes. For a single seed:

```bash
python scripts/train.py --config traisformer \
  --run-dir runs/traisformer-seed123 --seed 123 --device cuda:0

python scripts/train.py --config vessel_direct \
  --run-dir runs/b0-seed123 --seed 123 --device cuda:0

python scripts/train.py --config vessel_direct_b3a \
  --run-dir runs/b3a-seed123 --seed 123 --device cuda:0
```

For the paper-style three-seed protocol, repeat each command with seeds 123,
124, and 125 and a distinct run directory. No hyperparameter should change
between seeds.

Training writes:

```text
resolved_design.json          exact attempted configuration, written first
best_validation_state.pt      checkpoint selected by validation loss
validation_predictions.npz    standardized aligned predictions
validation_report.json        losses, timing, and shared metrics
vessel_history_context.npz    B3A training normalization/audit data only
```

B3A constructs its context internally. No separate preparation script is
needed.

## 5. Evaluate a trained run

Training already evaluates the selected checkpoint on validation. To rerun it
without overwriting the training artifacts, use a separate output directory:

```bash
python scripts/evaluate.py --config traisformer \
  --run-dir runs/traisformer-seed123 \
  --output-dir runs/recheck-traisformer-seed123 --split valid

python scripts/evaluate.py --config vessel_direct \
  --run-dir runs/b0-seed123 \
  --output-dir runs/recheck-b0-seed123 --split valid

python scripts/evaluate.py --config vessel_direct_b3a \
  --run-dir runs/b3a-seed123 \
  --output-dir runs/recheck-b3a-seed123 --split valid
```

The accepted corrected TrAISformer state preserves all original checkpoint
keys and can be evaluated without a new training run:

```bash
python scripts/evaluate.py --config traisformer \
  --checkpoint models/states/ct_dma/traisformer.pt \
  --output-dir runs/recheck-traisformer-reference --split valid
```

Its expected SHA-256 is recorded in `models/states/ct_dma/README.md`.

### Test evaluation

Test access is deliberately explicit:

```bash
python scripts/evaluate.py --config vessel_direct_b3a \
  --run-dir runs/b3a-seed123 \
  --output-dir runs/test-b3a-seed123 \
  --split test --confirm-frozen-design
```

For B3A, validation/test queries retrieve history from the training split only.
A query split never supplies context to itself. Do not change configurations,
retrieval rules, or checkpoints after inspecting test results.

## 6. Compare models

Every new evaluation artifact uses this model-independent layout:

```text
predictions     [examples, modes, future steps, lat/lon]
probabilities   [examples, modes]
clip_mask       [examples, modes, future steps]
targets         [examples, future steps, lat/lon]
target_mask     [examples, future steps]
trajectory_ids  [examples]
vessel_ids      [examples]
```

The comparison script asserts exact equality of trajectory IDs, targets, and
masks before calculating paired 1/2/3/6-hour FDE differences, win rate, and a
fixed-seed trajectory bootstrap confidence interval.

B0 versus B3A uses the deployed top-1 prediction (identical to expected error
for K=1):

```bash
python scripts/compare.py \
  runs/recheck-b3a-seed123/valid_predictions.npz \
  runs/recheck-b0-seed123/valid_predictions.npz \
  --prediction-rule top1
```

The corrected TrAISformer headline is the mean error across its 16 stochastic
rollouts. Use `expected`, not arbitrary mode 0, when comparing it with B0:

```bash
python scripts/compare.py \
  runs/recheck-b0-seed123/valid_predictions.npz \
  runs/recheck-traisformer-reference/valid_predictions.npz \
  --prediction-rule expected
```

Within JSON metric reports, the corresponding curves are
`top1_fde_nmi` for B0/B3A and `probability_weighted_fde_nmi` for the uniform
TrAISformer rollouts. Oracle/minFDE values are diagnostics, not deployable
predictions.

### Visual route comparison

To compare the actual predicted paths instead of aggregate errors, plot aligned
examples from the same two artifacts:

```bash
python scripts/plot_routes.py \
  runs/b0-seed123/validation_predictions.npz \
  runs/traisformer-seed123/validation_predictions.npz \
  --prediction-rule expected \
  --label-a "My model" --label-b "TrAISformer" \
  --output runs/b0-vs-traisformer-routes.png
```

The default figure contains nine representative examples spanning cases where
A is better, the models are similar, and B is better at six hours. The black
line is the true future route. The strong blue/orange lines are the models'
probability-weighted mean routes; faint lines show individual modes or
TrAISformer rollouts. A weighted mean route is a visual summary and can fall
between distinct candidate routes; it is not itself a sampled rollout.

For a repeatable random sample or hand-picked case studies, use:

```bash
python scripts/plot_routes.py ARTIFACT_A ARTIFACT_B \
  --selection random --num-examples 12 --seed 123 --output routes.pdf

python scripts/plot_routes.py ARTIFACT_A ARTIFACT_B \
  --trajectory-ids 12345 23456 34567 --output selected-routes.svg
```

The plotting command requires Matplotlib in addition to the training/evaluation
environment.

Historical artifacts created before this refactor use a mode-major layout.
Keep them as evidence, but produce fresh comparison artifacts with
`scripts/evaluate.py` rather than feeding old files directly to the new paired
comparison script.

## 7. B3A vessel-history mechanism

B3A searches only the permitted training-source trajectories. A match must
have the same MMSI and the same one-of-eight final COG sector. Its six-hour
endpoint timestamp must be strictly earlier than the query forecast origin.
Among eligible routes, B3A selects the nearest physical forecast origin.

The context is:

```text
[E3, N3, E6, N6, match_distance, log1p(pool_size), support]
```

Endpoint displacements are relative to the matched route's own origin and use
nautical miles. Continuous values are normalized using supported training rows
only. Unsupported queries receive six zeros and `support=0`.

## 8. Repository structure

```text
configs/dataset/          dataset profiles and split locations
configs/model/            one config per executable model
forecasting/data.py       common loading, target extraction, dataset views
forecasting/adapters.py   genuine model-specific loss/prediction differences
forecasting/runner.py     shared train/checkpoint/evaluate/artifact lifecycle
evaluation/metrics.py     shared geographic and probabilistic metrics
models/baselines/         executable paper baselines, currently TrAISformer
models/direct/            B0/B3A, physical decoder, and vessel history
preprocessing/            retained Danish AIS preprocessing
scripts/train.py          shared trainer
scripts/evaluate.py       shared evaluator
scripts/compare.py        paired model comparison
tests/                    framework, numerical, and leakage invariants
```

There is intentionally no `pipelines/` package anymore. Its useful behavior is
now divided by responsibility:

- `forecasting/runner.py` is the single experiment pipeline;
- `forecasting/data.py` owns common data and targets;
- `forecasting/adapters.py` isolates unavoidable model differences;
- `evaluation/metrics.py` evaluates every model identically.

To add another baseline or proposed model, add its model/config, implement the
small adapter contract, and register it in `build_adapter`. It then inherits
the same split handling, runner, artifacts, metrics, gating, and comparisons.

## 9. Preserved research results

`RESEARCH_HISTORY.md` is the authoritative completed-cycle research record.
The historical `runs/` artifacts remain evidence and are not regenerated by
this refactor.

Numerical compatibility checks for the new structure found:

- B0 validation prediction maximum difference: `4.77e-07`;
- B3A validation prediction maximum difference: `2.38e-07`;
- TrAISformer old/new forward-logit maximum difference: `0.0`;
- TrAISformer old/new training-loss difference: `0.0`;
- full 16-rollout TrAISformer validation replay: 850,175 of 850,176
  predicted coordinates exactly identical, with one categorical sample one bin
  different; headline 1/2/3/6-hour mean FDE values differ by less than
  `1.1e-7` nmi and the maximum difference anywhere on the 72-step mean curve is
  `0.000111` nmi;
- accepted TrAISformer, B0, and B3A checkpoints load strictly.

The very small B0/B3A prediction differences are floating-point noise from
execution, not changed algorithms. Prediction artifacts now use an
example-major layout and include extra alignment fields, so old and new `.npz`
files are not expected to be byte-identical even when their numerical forecasts
match.
