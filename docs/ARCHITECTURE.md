# Architecture and Extension Points

The repository follows a simple dependency direction:

```text
raw dataset
    ↓
src/ids/data/              dataset-specific parsing / label normalization
    ↓
src/ids/preprocessing/     split, common features, selector, scaler
    ↓
model-facing arrays
    ↓
src/ids/models/            neural or classical model only
    ↓
src/ids/evaluation/        metrics
    ↓
src/ids/experiments/       logging / run metadata
    ↓
results/summary.csv + run directory
```

## Data input boundary

Dataset-specific assumptions belong in `src/ids/data/`. Examples:

- CICIoT2023 fine label -> binary/category label;
- UNSW-NB15 official split detection;
- CIC-IDS2018 historical 40k/class construction.

After adaptation, processed datasets should expose a clear `Label` column and numeric
features suitable for an explicit preprocessing step.

## Preprocessing boundary

Preprocessing code decides:

- which samples are train/validation/test;
- how domains are aligned;
- which feature names are used and in what order;
- where feature selectors/scalers are fit.

Models must not make these decisions themselves.

## Model input boundary

For clean UDA, the conceptual contract is:

```text
source: X_train, y_train, X_val, y_val
target: X_unlabeled
final evaluation only: X_target_test, y_target_test
```

This contract prevents accidental target-label leakage in DANN, PACDA-IDS, CORAL,
MMD-based methods, or future domain adaptation models.

## Output boundary

A script should report metrics through `ExperimentTracker.record_metrics()` rather than
invent a new result file. This guarantees a common `results/summary.csv` for plotting
and aggregation.

## Adding a model

1. implement it under `src/ids/models/`;
2. optionally expose the builder in `src/ids/models/registry.py`;
3. create/configure an experiment script;
4. receive prepared arrays, not file paths;
5. log history, model summary, metrics and feature order.

## Adding a dataset

1. register it in `configs/datasets.yaml`;
2. implement an adapter under `src/ids/data/`;
3. create `scripts/prepare_<name>.py`;
4. inspect class counts before training;
5. document source, version and label mapping in `docs/DATASETS.md`.
