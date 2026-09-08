# DANN fairness control — v0.4.7

v0.4.6 established that the audited MLP-DANN domain head was strongly reversed while a frozen probe still separated source from target almost perfectly. That is **not successful alignment**. Before interpreting the result as negative transfer, v0.4.7 removes two confounds present in the v0.4.5/v0.4.6 MLP comparison.

## What was confounded

1. Source-only MLP and MLP-DANN did **not** have the same task head. The DANN model had an additional hidden classification layer and dropout.
2. Source-only MLP used a learning rate of `1e-3`, while DANN defaulted to `1e-4`.
3. The original MLP used BatchNorm, whose moving statistics can themselves be affected by heterogeneous source/target batches.

Therefore the old MLP vs MLP-DANN diagnostic is useful for debugging, but it is not a fair scientific ablation.

## v0.4.7 control

Use `--encoder mlp_fair`.

Both source-only and DANN now share exactly:

```text
12 canonical features
  -> Dense(64) + ReLU
  -> LayerNorm
  -> Dropout
  -> Dense(32) + ReLU   [embedding]
  -> Dense(32) + ReLU
  -> Dropout
  -> 2-class task head
```

DANN only adds:

```text
embedding -> GRL -> Dense(32) -> 2-class domain head
```

For MLP controls, if `--dann-lr` is omitted, DANN automatically uses the same learning rate as `--source-lr` (default `1e-3`). Clean DANN training also uses one mixed source+target forward pass per step. Target class labels remain absent.

## Mandatory lambda=0 control

First verify that adding an inert domain branch does not destroy the task model:

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

Expected: `MLP-Fair-Canonical-Control` and `MLP-Fair-DANN-Canonical-Control` should be reasonably close. They need not be bit-identical because the training loops differ, but a catastrophic ranking inversion at `lambda_d=0` is a pipeline bug/optimization confound, not DANN evidence.

Then run the actual adversarial smoke:

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

Only after the lambda=0 control behaves sensibly may the `lambda_d=0.1` result be interpreted as negative transfer or successful adaptation.
