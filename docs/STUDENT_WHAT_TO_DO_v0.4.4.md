# STUDENT WHAT TO DO — IDS v0.4.4

## Why v0.4.4
The v0.4.3 DANN smoke logged `best_source_val_macro_f1=0.0999`, but after training the same run selected source threshold `0.195` with source-val metric `0.5908`. The training checkpoint had been judged at fixed threshold 0.5. v0.4.4 fixes this inconsistency.

## Do now
Do **one** decisive smoke run only. Do not run full DANN, seed sweeps, PACDA, FGSM, or PGD yet.

```powershell
python scripts/run_transfer_clean.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --only all `
  --lambda-d 0.1 `
  --target-threshold-strategy gmm_logit `
  --smoke
```

## What the log must show
DANN epochs must print `val_thr=...`, `target_pos@0.5=...`, and `target_pos@src_thr=...`. Checkpointing is based on threshold-aware source-val Macro-F1.

## Report back
Send the final two-row table for Source-only and Clean-DANN plus: `best_source_val_macro_f1`, source threshold, GMM threshold status, AUROC, Macro-F1, Balanced Accuracy, FPR, FNR, and runtime.

## Decision rule
- Source-only healthy, DANN worse: negative transfer candidate.
- Both healthy on source, target AUROC poor: cross-domain neural representation problem.
- Target AUROC good, Macro-F1 poor: calibration problem.
- Any `UNVERIFIED_DIAGNOSTIC` run is engineering evidence only, not a paper result.
