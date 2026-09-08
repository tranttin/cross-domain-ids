# IDS Journal Experiment Runbook — v0.5.0

The paper experiment plan is now configuration-driven. The single source of truth is:

- `configs/paper_experiments_v050.yaml`
- orchestration entry point: `scripts/run_paper.py`
- paper-table exporter: `scripts/build_paper_tables.py`

Do not edit individual scripts to change paper settings. Change the YAML only after the protocol is deliberately re-frozen.

## 0. Before any final run

### Environment

```powershell
python -c "import sys, tensorflow as tf; print(sys.executable); print(tf.__file__)"
python scripts/check_paper_protocol.py --config configs/paper_experiments_v050.yaml --final
```

The gate must not report TensorFlow from `LightGCN\venv`.

### Freeze a fresh CICIoT2023 DEV / FINAL_TEST pool

Do this once, from a fresh prepared pool that is not the earlier `ciciot2023_sample.csv` tuning file.

```powershell
python scripts/prepare_ciciot2023.py `
  --raw-dir data/raw/ciciot2023 `
  --output data/processed/ciciot2023_protocol_pool.csv `
  --sample-frac 0.05 `
  --task binary `
  --seed 20260819

python scripts/freeze_ciciot2023_split.py `
  --input data/processed/ciciot2023_protocol_pool.csv `
  --dev-output data/processed/ciciot2023_dev.csv `
  --test-output data/processed/ciciot2023_final_test.csv `
  --seed 20260819
```

Never overwrite this split after looking at final-test results.

### Canonical mapping gate

`configs/features/canonical_features.yaml` still marks the CICIoT2023 semantic mappings as unverified in v0.5.0. Final E3/E4 is therefore intentionally blocked until those exact prepared columns/units are verified and the entries are changed to `verified: true`.

---

## E1 -> Paper Table T2: In-domain benchmark

Models: NB, SGD, LR, DT, RF, CNN-BiLSTM-Attention.

Smoke:

```powershell
python scripts/run_paper.py --stage E1 --mode smoke
```

Final, seeds 42/43/44 from YAML:

```powershell
python scripts/run_paper.py --stage E1 --mode final
```

Manual single run:

```powershell
python scripts/run_in_domain_clean.py `
  --csv data/processed/cleaned_data_sampled.csv `
  --dataset-name 2018 `
  --seed 42
```

IoT final uses the frozen test explicitly:

```powershell
python scripts/run_in_domain_clean.py `
  --csv data/processed/ciciot2023_dev.csv `
  --test-csv data/processed/ciciot2023_final_test.csv `
  --dataset-name CICIoT2023 `
  --seed 42
```

---

## E2 -> Paper Table T3: CIC related-domain cross transfer

Directions: 2017->2018 and 2018->2017.

Rows kept in paper: LR, RF, CNN source-only, clean DANN.

Smoke:

```powershell
python scripts/run_paper.py --stage E2 --mode smoke
```

Final:

```powershell
python scripts/run_paper.py --stage E2 --mode final
```

The orchestrator fixes `lambda_d=0.01` from the paper YAML for clean DANN and uses threshold-aware source-validation checkpointing.

---

## E3 -> Paper Table T4: heterogeneous transfer to CICIoT2023

Directions: 2017->IoT2023 and 2018->IoT2023.

Rows: LR, RF, MLP-BN canonical source-only, MLP-BN + staged DANN.

Smoke/diagnostic:

```powershell
python scripts/run_paper.py --stage E3 --mode smoke
```

Final after E0 gate passes:

```powershell
python scripts/run_paper.py --stage E3 --mode final
```

Manual 2018->IoT final:

```powershell
python scripts/run_transfer_clean.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_dev.csv `
  --target-test-csv data/processed/ciciot2023_final_test.csv `
  --target-name CICIoT2023 `
  --encoder mlp_bn_fair `
  --only all `
  --lambda-d 0.01 `
  --target-threshold-strategy gmm_logit `
  --seed 42
```

---

## E4 -> Paper Table T5: multi-source unseen-IoT transfer

Source: 2017+2018. Target: CICIoT2023.

Smoke:

```powershell
python scripts/run_paper.py --stage E4 --mode smoke
```

Final:

```powershell
python scripts/run_paper.py --stage E4 --mode final
```

Rows: MLP-BN canonical and MLP-BN + DANN (lambda 0.01).

---

## E5 -> Paper Figure F3: DANN lambda sensitivity

Development ablation only. It deliberately uses the diagnostic IoT sample and never the frozen final test.

```powershell
python scripts/run_paper.py --stage E5 --mode smoke
```

Lambda grid from YAML:

```text
0, 0.001, 0.005, 0.01, 0.05, 0.1
```

This is not a main baseline table. It is a hyperparameter-sensitivity/negative-transfer ablation.

---

## E6 -> Paper Table T6 / Figure F4: FGSM and PGD robustness

v0.5.0 provides a diagnostic attack runner for pipeline validation:

```powershell
python scripts/run_paper.py --stage E6 --mode smoke
```

It evaluates FGSM and PGD at:

```text
0, 0.01, 0.03, 0.05, 0.10
```

**Important:** v0.5.0 attacks are still in scaled model-input space and are marked `UNCONSTRAINED_DIAGNOSTIC`. `--mode final` is intentionally blocked. After manuscript review, define the exact mutability/bounds/integer constraints and only then enable final robustness evidence and adversarial-training variants.

---

## Build paper tables automatically

After any set of runs:

```powershell
python scripts/build_paper_tables.py `
  --summary results/summary.csv `
  --output-dir results/paper_tables `
  --lambda-main 0.01
```

Outputs:

```text
results/paper_tables/T2_in_domain_mean_std.csv
results/paper_tables/T3_CIC_cross_domain_mean_std.csv
results/paper_tables/T4_heterogeneous_transfer_mean_std.csv
results/paper_tables/T5_multisource_mean_std.csv
results/paper_tables/F3_lambda_sensitivity_rows.csv
results/paper_tables/T6_robustness_mean_std.csv
```

## Dry-run before spending compute

Always inspect the exact commands first:

```powershell
python scripts/run_paper.py --stage E3 --mode final --dry-run
```

For one final seed only:

```powershell
python scripts/run_paper.py --stage E3 --mode final --seed 42 --dry-run
python scripts/run_paper.py --stage E3 --mode final --seed 42
```

## Recommended execution order

```text
E0 gate
 -> E1 smoke
 -> E2 smoke
 -> E3 smoke
 -> E4 smoke
 -> build tables / inspect
 -> E1 final seed42
 -> E2 final seed42
 -> E3 final seed42
 -> E4 final seed42
 -> inspect
 -> final seeds 43/44
 -> E6 constrained robustness after manuscript method is frozen
```

E5 lambda sensitivity has already served its development purpose. Do not keep tuning lambda after the final test is frozen.
