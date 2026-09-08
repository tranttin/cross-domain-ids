# STUDENT WHAT TO DO — v0.4.5

## Goal of this version

Run one controlled experiment to determine whether the poor CICIoT2023 transfer comes
from the sequence-style CNN--BiLSTM--Attention encoder or from the canonical feature
space itself. Do **not** start PACDA, FGSM/PGD, seed sweeps, or full DANN tuning yet.

## Step 0 — environment check

```powershell
python scripts/check_environment.py
python -c "import sys, tensorflow as tf; print(sys.executable); print(tf.__file__)"
```

Both paths must belong to the IDS-2026 environment. If TensorFlow is imported from a
LightGCN environment, fix that before any final experiment. A smoke run may be used only
for engineering diagnosis.

## Step 1 — run the MLP control smoke

```powershell
python scripts/run_transfer_clean.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --encoder mlp `
  --only all `
  --lambda-d 0.1 `
  --target-threshold-strategy gmm_logit `
  --smoke
```

This produces two rows:

- `MLP-Canonical-Control`
- `MLP-DANN-Canonical-Control`

## Step 2 — report these fields

For both rows, copy:

- `best_source_val_macro_f1`
- `source_val_decision_threshold`
- `decision_threshold`
- `threshold_status`
- `target_adapt_positive_rate_at_threshold`
- `auroc`
- `auprc` and `auprc_baseline`
- `macro_f1`
- `balanced_accuracy`
- `fpr`, `fnr`
- `train_seconds`

For MLP+DANN also report:

- `final_val_domain_accuracy`
- `final_unlabelled_target_predicted_positive_rate`

## Step 3 — decision rule

1. **MLP target AUROC >= 0.75 and much higher than CNN smoke**: encoder mismatch is supported.
2. **MLP+DANN >= MLP on target AUROC and balanced metrics**: DANN helps under a compatible encoder.
3. **MLP good, MLP+DANN worse**: vanilla DANN negative transfer is supported.
4. **Both near random**: stop deep runs and revisit harmonization/preprocessing.

These thresholds are engineering gates, not publication significance tests.

## Forbidden for now

- Do not use target-test labels to select threshold or checkpoint.
- Do not choose a feature subset from target-test ablation as the final paper mapping.
- Do not report `--smoke` or `--allow-unverified-mapping` rows as final paper results.
- Do not run multiple seeds until the single-seed architecture-control question is answered.

## What to send to the supervisor

```text
Git commit:
Command:
Run ID:
Encoder:
Mapping version:
Threshold strategy:
MLP metrics:
MLP+DANN metrics:
Runtime:
Conclusion in one sentence:
```
