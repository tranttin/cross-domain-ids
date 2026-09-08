# IDS-2026 v0.4.3 — What Students Must Do Next

**Current phase:** 2018 → CICIoT2023 cross-domain debugging and clean calibration.  
**Do not start PACDA, FGSM/PGD, multi-seed, or full DANN runs yet.**

## Rule 0 — Do not change the scientific protocol

Students may run scripts, inspect outputs, document findings, and fix confirmed implementation bugs. Do **not** silently change feature mappings, class labels, thresholds, train/test splits, loss weights, or model architecture.

`legacy/` is frozen. New research changes belong only to the clean research path and must be committed separately.

## Task 1 — Verify environment

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/check_environment.py
```

If TensorFlow resolves from another project/venv, stop and fix the environment before running experiments.

## Task 2 — Run feature-stability audit

```powershell
python scripts/audit_feature_stability.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping
```

Submit:

- console output;
- `results/diagnostics/feature_stability_2018_to_CICIoT2023/feature_stability.csv`;
- a 5–10 line note identifying large coefficient, large source-validation LOO drop, and large unlabelled target shift features.

Do **not** remove a feature just because target-test diagnostic AUROC improved in an older ablation.

## Task 3 — Run label-free calibration

```powershell
python scripts/run_unsupervised_calibration.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --model all
```

Submit:

- `calibration_results.csv`;
- `threshold_details.json`;
- `PROTOCOL.json`;
- a short comparison of `source`, `gmm_logit`, `valley_logit`, and `source_prior_quantile`.

The **oracle** columns are diagnostic upper bounds only. Never call them clean-UDA results.

## Task 4 — Deep smoke only after Tasks 2–3

Run source-only twice:

```powershell
python scripts/run_transfer_clean.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --only source_only `
  --target-threshold-strategy source `
  --smoke
```

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

Then, only if those runs complete without collapse, run DANN smoke:

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

## Stop conditions

Stop and report immediately if any of these occur:

- mapped/selected features unexpectedly change;
- NaN/Inf appears;
- predicted-positive rate collapses to approximately 0 or 1;
- GMM reports a tiny/weak component and produces a pathological threshold;
- TensorFlow loads from the wrong environment;
- a run needs target-test labels for training or threshold selection.

Do not “fix” a bad result by manually changing the target threshold.

## What to report for every run

Copy these into the student experiment log:

```text
Date/time:
Git commit:
Command:
Run ID:
Source → Target:
Mapping version:
Selected feature count:
Model:
Threshold strategy:
Threshold fit pool:
Macro-F1:
Balanced Accuracy:
AUPRC:
AUROC:
FPR:
FNR:
Predicted-positive rate:
Runtime:
Observation (3–5 lines):
```

Attach the corresponding `results/runs/<run_id>/` or diagnostic output directory. Never report a metric without its run ID.

## Not allowed yet

Until the supervisor explicitly approves the smoke results, do not run:

```text
seed 43/44/...
full 30-epoch CICIoT2023 DANN
lambda sweep
PACDA
adversarial training
FGSM / PGD
final manuscript tables
```

The immediate objective is to establish a leakage-free, stable 2018 → CICIoT2023 transfer/calibration baseline first.
