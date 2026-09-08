# Student Guide

## Goal

This repository is both a reproduction package and a research codebase. Your job is
not merely to obtain a high score. Your job is to produce a result that can be traced
from **dataset -> split -> preprocessing -> model -> checkpoint -> evaluation -> table**.

## First-day setup

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_environment.py
kaggle auth login
python scripts/download_data.py --dataset legacy
python scripts/inspect_dataset.py --all
```

If `python scripts/check_environment.py` points to another project's environment,
stop and fix the venv before running experiments.

## Required reproduction milestones

Run in this order:

```powershell
python scripts/run_baselines_2018.py
python scripts/run_hybrid_2018.py
python scripts/run_cross_domain_legacy.py --cross-only
python scripts/compare_legacy_reference.py
```

Keep the generated run directories. Do not report only screenshots.

## Run artefact checklist

A deep-learning run should normally contain:

```text
results/runs/<run_id>/
├── config.yaml
├── metadata.json
├── run.log
├── metrics.csv
├── table.csv               # if the script evaluates multiple rows
├── history*.csv
├── feature*.json/txt
├── model_summary*.txt
└── predictions.csv         # where practical
```

`metadata.json` records Python/package versions, command line and duration. This is
important because TensorFlow, scikit-learn and Python version changes can affect results.

## Weekly research log

For every meaningful run, record:

- run_id;
- Git commit hash;
- experiment question;
- source/target dataset;
- model/config changes;
- seed(s);
- validation criterion;
- final metrics;
- failure/observation;
- next action.

Do not write "changed model and accuracy improved". Write exactly what changed.

## Adding a new dataset

1. Add metadata to `configs/datasets.yaml`.
2. Add an adapter under `src/ids/data/` if labels/features need normalization.
3. Write a `scripts/prepare_<dataset>.py` entry point.
4. Ensure the processed file exposes a clear `Label` column.
5. Log binary/multiclass label semantics.
6. Never force two datasets to share features by silently renaming unrelated columns.
7. For cross-domain experiments, report the common feature set and its order.

## Adding a new model

Put model construction under `src/ids/models/`. Model code should not:

- read CSV files;
- create train/test splits;
- fit scalers/selectors;
- access target labels during UDA training;
- write results to arbitrary locations.

The experiment script owns data selection and logging.

## Adding PACDA-IDS or another adaptation method

Use the clean source/target split from `prepare_clean_source_target()` as the data contract.
The method should receive:

```text
X_source_train, y_source_train
X_source_val, y_source_val
X_target_unlabeled
```

It must not receive `y_target_test` until final evaluation. This makes comparisons
against clean DANN defensible.

## Repeated experiments

A single deep run is useful for debugging but insufficient for a final paper claim.
Once a configuration is frozen, run multiple seeds and aggregate mean/std from
`results/summary.csv`. Do not tune on the final target test set.

## If a long run is interrupted

Do not restart a four-job tour blindly. Narrow the job:

```powershell
python scripts/run_cross_domain_clean.py --direction 2018-2017 --only dann
```

Or use global-summary resume support:

```powershell
python scripts/run_cross_domain_clean.py --resume
```

Before final experiments, `python scripts/check_environment.py` must report that the
TensorFlow module is loaded from the same virtual environment as `sys.prefix`. If it
prints `TF in this venv: NO`, fix the PyCharm interpreter/PYTHONPATH before generating
paper results.

## v0.4 mandatory third-domain checklist

Do not launch a long CICIoT2023 DANN run immediately after preparing the CSV.
Follow this order and keep the generated artefacts in `results/`:

1. `prepare_ciciot2023.py` and check the `.meta.json` class counts.
2. `diagnose_domain_shift.py --allow-unverified-mapping`.
3. Open `feature_mapping.csv`; verify definitions/units in the dataset documentation.
4. Mark only justified mappings as `verified: true` in `configs/features/canonical_features.yaml`.
5. `run_transfer_baselines.py` (LR/RF/SGD).
6. `run_transfer_clean.py --only source_only --smoke`.
7. `run_transfer_clean.py --only dann --lambda-d 0.1 --smoke`.
8. Only after the above is sensible, remove diagnostic flags and run full seed 42.
9. Multi-seed runs come last.

A run with `--allow-unverified-mapping` or `--smoke` is **not a paper result**.
If all cheap baselines are around AUROC/Balanced Accuracy 0.5, report the mapping/data
problem rather than tuning the neural network for hours.

## v0.4.2 third-domain rule: separate ranking from threshold transfer

If a cheap baseline has high AUROC/AUPRC but Macro-F1 or Balanced Accuracy collapses,
do not immediately tune DANN. Run `scripts/diagnose_score_shift.py` and inspect whether
target scores preserve class ranking but no longer cross the threshold chosen on source
validation. Target-labelled oracle thresholds and mapping ablations are **diagnostic
only** and may not be used to select the final clean-UDA threshold or feature set.

Recommended order for 2018 -> CICIoT2023:

```text
diagnose_domain_shift.py
    -> run_transfer_baselines.py
    -> diagnose_score_shift.py (when gate says SCORE-SHIFT-SUSPECT/MIXED-REVIEW)
    -> run_mapping_ablation.py --mode groups (semantic debugging only)
    -> verify mapping semantics/units
    -> source-only deep smoke
    -> DANN smoke
```

## v0.4.3 immediate assignment

For the current 2018 → CICIoT2023 phase, follow `docs/STUDENT_WHAT_TO_DO_v0.4.3.md` exactly. Do not start full DANN, multi-seed, PACDA, adversarial training, FGSM or PGD until the v0.4.3 feature-stability and label-free calibration diagnostics have been reviewed.
