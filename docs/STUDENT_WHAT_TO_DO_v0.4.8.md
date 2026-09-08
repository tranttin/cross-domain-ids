# STUDENT WHAT TO DO — v0.4.8

Do not run full DANN, seed sweeps, PACDA, FGSM, or PGD yet.

## 1. Check environment
```powershell
python scripts/check_environment.py
python -c "import sys, tensorflow as tf; print(sys.executable); print(tf.__file__)"
```
Both paths must belong to `IDS-2026`. If TensorFlow still comes from `LightGCN\venv`, stop before final experiments.

## 2. Run the staged fairness control
```powershell
python scripts/run_dann_staged_control.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --target-threshold-strategy gmm_logit `
  --smoke
```

This runs `lambda=0` and `lambda=0.1` with the same source-pretrained BatchNorm MLP and frozen BatchNorm statistics during adaptation.

## 3. Report
For source-only, lambda=0, and lambda=0.1 report: `best_source_val_macro_f1`, target `AUROC`, `AUPRC`, `Macro-F1`, `Balanced Accuracy`, `FPR`, `FNR`, `domain balanced accuracy`, threshold status, run_id, and runtime.

Do not select a threshold or feature subset from target-test labels.
