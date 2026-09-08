# STUDENT WHAT TO DO — v0.4.6

## Only task now

Audit the MLP+DANN domain head. Do **not** run full DANN, lambda sweeps, PACDA, FGSM/PGD,
or multiple seeds yet.

### Step 0 — environment

```powershell
python scripts/check_environment.py
python -c "import sys, tensorflow as tf; print(sys.executable); print(tf.__file__)"
```

For final experiments both paths must belong to the IDS-2026 environment. If the current
smoke still imports TensorFlow from `LightGCN\venv`, mark the result engineering-only.

### Step 1 — run the audit

```powershell
python scripts/audit_dann_domain_head.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --encoder mlp `
  --lambda-d 0.1 `
  --epochs 4
```

### Step 2 — send these fields

```text
orientation_status:
domain_head_balanced_accuracy:
domain_head_reversed_balanced_accuracy:
domain_head_AUROC:
domain_head_reversed_AUROC:
source_mean_P_target:
target_mean_P_target:
canonical_input_probe_AUROC:
embedding_probe_AUROC:
embedding_minus_input_AUROC:
best_source_val_macro_f1:
```

Also attach `results/diagnostics/dann_domain_head_2018_to_CICIoT2023/domain_head_audit.json`.

## Decision rule

- `embedding_probe_AUROC ~ 0.5`: DANN genuinely removed much domain information; investigate why task semantics were destroyed.
- `embedding_probe_AUROC high (>=0.8)`: DANN did **not** create domain invariance even if its own head accuracy is low; treat low head accuracy as misleading.
- If the head is reversed but the embedding probe is high, do not flip labels and call it fixed. The representation is still domain-separable.

Target attack/benign labels must not be used in this audit.
