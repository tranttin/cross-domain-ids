# v0.4.3 — Feature Stability and Label-Free Target Calibration

## Why this stage exists

The 2018 → CICIoT2023 diagnostic currently shows a useful but dangerous pattern:

- the 13-feature candidate semantic map contains strong cross-domain ranking signal;
- Logistic Regression reached target AUROC around 0.958 in diagnostic evaluation;
- the source-validation threshold collapsed on the target score scale, so hard-label Macro-F1 was near zero;
- an oracle target threshold can recover much of the lost performance, but it uses target-test labels and is **diagnostic only**.

Therefore v0.4.3 separates **representation/ranking quality** from **decision-threshold transfer**.

## Non-negotiable protocol rule

Target data are split into:

1. `target_unlabelled_adaptation` — labels are unavailable to the method; label-free calibration may use scores from this pool;
2. `target_test` — labels are held out and used only after the threshold is frozen, for final evaluation.

No threshold strategy may inspect `y_target_test`.

## Feature-stability audit

Run:

```powershell
python scripts/audit_feature_stability.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping
```

Outputs include:

- balanced-LR coefficient per selected canonical feature;
- source-validation AUROC/Macro-F1 drop when each feature is removed;
- source-vs-unlabelled-target scaled median/p01/p99 shift;
- target fraction outside source p01–p99;
- change in median absolute LR contribution.

This script is **source-led**: target-test labels are not used to rank or choose features.

## Label-free target threshold strategies

`run_unsupervised_calibration.py` evaluates four pre-specified strategies:

- `source` — frozen threshold selected on labelled source validation;
- `gmm_logit` — two-component Gaussian mixture fitted to target-adaptation logit scores;
- `valley_logit` — robust largest-gap threshold on target-adaptation logit scores;
- `source_prior_quantile` — control that matches predicted target-adaptation positive fraction to labelled source-train prevalence.

Run:

```powershell
python scripts/run_unsupervised_calibration.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --model all
```

The candidate thresholds are fitted from `X_target_unlabeled` scores only. The script then freezes each threshold and evaluates it once on held-out target test labels.

### Oracle warning

`diagnostic_oracle_target_threshold` and `diagnostic_oracle_target_macro_f1` are upper-bound diagnostics. They are forbidden for:

- model selection;
- hyperparameter tuning;
- mapping selection;
- final deployable thresholding;
- claims of clean UDA performance.

## Deep smoke after calibration diagnostics

The generic clean runner now supports:

```text
--target-threshold-strategy source
--target-threshold-strategy gmm_logit
--target-threshold-strategy valley_logit
--target-threshold-strategy source_prior_quantile
```

Example source-only smoke:

```powershell
python scripts/run_transfer_clean.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --only source_only `
  --target-threshold-strategy gmm_logit `
  --smoke
```

Example DANN smoke:

```powershell
python scripts/run_transfer_clean.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --only dann `
  --lambda-d 0.1 `
  --target-threshold-strategy gmm_logit `
  --smoke
```

Do not run full deep experiments or multi-seed experiments until the smoke runs are reviewed.
