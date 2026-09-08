# WHAT TO DO — v0.4.7

## Goal

Do **not** run full DANN, seed sweeps, PACDA, FGSM, or PGD yet. The current task is to establish an apples-to-apples MLP vs MLP+DANN control on 2018 -> CICIoT2023.

## Step 0 — environment

```powershell
python scripts/check_environment.py
python -c "import sys, tensorflow as tf; print(sys.executable); print(tf.__file__)"
```

Both Python and TensorFlow must come from the IDS-2026 environment. If TensorFlow still resolves to `LightGCN\\venv`, report it before final experiments.

## Step 1 — lambda=0 fairness smoke (mandatory)

```powershell
python scripts/run_transfer_clean.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --encoder mlp_fair `
  --only all `
  --lambda-d 0.0 `
  --target-threshold-strategy gmm_logit `
  --smoke
```

Send the full two-row table. Compare especially: `best_source_val_macro_f1`, `auroc`, `auprc`, `macro_f1`, `balanced_accuracy`, `fpr`, `fnr`, and threshold status.

## Step 2 — lambda=0.1 smoke

Run this only if lambda=0 does not catastrophically diverge from source-only.

```powershell
python scripts/run_transfer_clean.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --encoder mlp_fair `
  --only dann `
  --lambda-d 0.1 `
  --target-threshold-strategy gmm_logit `
  --smoke
```

## Step 3 — domain-head audit if needed

```powershell
python scripts/audit_dann_domain_head.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --encoder mlp_fair `
  --lambda-d 0.1 `
  --epochs 8
```

## Reporting format

Every student report must include Git commit, exact command, run_id, encoder, lambda_d, effective DANN LR, mapping version, threshold strategy, metrics, runtime, and whether the mapping is verified. Smoke/unverified results are diagnostics only.
