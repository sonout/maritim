# Training and evaluation

Run commands from the repository root after completing the [quick start](../README.md#quick-start).

## Quick preflight

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

## Train all canonical systems

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

## Evaluate a trained run

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

If you already have the local corrected TrAISformer checkpoint, it can be
evaluated without a new training run. This binary is not included in the repository:

```bash
python scripts/evaluate.py --config traisformer \
  --checkpoint models/states/ct_dma/traisformer.pt \
  --output-dir runs/recheck-traisformer-reference --split valid
```

The historical local reference checkpoint SHA-256 is
`227a7cf42d7ab587d914172c2bba259e2c246d634e8e4f7b179e925c34b2b0e2`.

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

## Compare models

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
  --label-a "B0" --label-b "TrAISformer" \
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

Matplotlib is included in the single root `requirements.txt`. Regenerate the
synthetic README illustration with `python scripts/plot_demo.py`.

Historical artifacts created before this refactor use a mode-major layout.
Keep them as evidence, but produce fresh comparison artifacts with
`scripts/evaluate.py` rather than feeding old files directly to the new paired
comparison script.

## B3A vessel-history mechanism

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
