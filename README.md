# Cross-Domain Intrusion Detection

Reproducible research code for:

**Domain-Adaptive Intrusion Detection across Heterogeneous Enterprise and IoT Networks**

Nguyen T. Phong, Pham Thi Thuy An, Bui Huong Giang, and Tin T. Tran.

This repository contains the experimental pipeline supporting the study of cross-domain network intrusion detection across enterprise and IoT environments.

The work evaluates intrusion detection under distribution shift rather than only under conventional in-domain train/test settings. Experiments use CIC-IDS2017, CSE-CIC-IDS2018, and CICIoT2023 and examine classical machine learning, boosted-tree models, CNN-BiLSTM-Attention, canonical tabular models, Domain-Adversarial Neural Networks (DANN), heterogeneous transfer, multi-source transfer, class-prior and threshold shift, and constraint-aware adversarial robustness.

---

## Research scope

The study addresses three main questions:

1. How well do classical, boosted-tree, and deep IDS models perform under in-domain evaluation?
2. How well do IDS models transfer across related enterprise domains and across heterogeneous enterprise-to-IoT domains?
3. How robust are source-only and domain-adapted canonical IDS models to constraint-aware feature-space FGSM and PGD perturbations?

A central finding is that strong in-domain performance does not guarantee cross-domain robustness. XGBoost and CatBoost are strong tabular baselines, while the benefit of DANN is strongly dependent on the source-target direction and representation compatibility.

---

## Datasets

The experiments use three public network-security datasets:

| Dataset | Environment | Main role |
|---|---|---|
| CIC-IDS2017 | Enterprise-style network | In-domain and cross-domain source/target |
| CSE-CIC-IDS2018 | Enterprise-style network | In-domain and cross-domain source/target |
| CICIoT2023 | IoT network | Heterogeneous target and robustness evaluation |

The datasets themselves are **not distributed in this repository**.

Dataset definitions and download/preparation information are provided in:

```text
configs/datasets.yaml
```

Typical prepared files include:

```text
data/processed/cicids2017_balanced_77feats.csv
data/processed/cleaned_data_sampled.csv
data/processed/ciciot2023_sample.csv
data/processed/ciciot2023_dev.csv
data/processed/ciciot2023_final_test.csv
```

---

## Heterogeneous feature harmonization

CIC-IDS2017 and CSE-CIC-IDS2018 use related CICFlowMeter-style feature representations, whereas CICIoT2023 exposes a different native schema.

Enterprise-to-IoT experiments therefore do not rely on raw column-name overlap. They use a semantically audited canonical representation.

The final strict canonical core contains seven variables:

```text
flow_packets_per_second
forward_packets_per_second
backward_packets_per_second
packet_length_min
packet_length_mean
packet_length_max
packet_length_std
```

The complete mapping and excluded candidate variables are documented in:

```text
configs/features/canonical_features.yaml
```

---

## Representation strategy

The evaluation uses two representation paths:

- **CIC-IDS2017 <-> CSE-CIC-IDS2018:** CNN-BiLSTM-Attention for the related CIC feature representation.
- **Enterprise -> CICIoT2023:** a strict seven-feature canonical representation with a lightweight tabular MLP.

DANN is an optional adaptation component rather than a separate prediction architecture. The attack classifier is trained using labeled source data, while unlabeled target data contribute to the domain-adversarial objective.

The CNN-BiLSTM-Attention path operates on an ordered within-record feature representation. It does **not** construct chronological multi-flow sequences, and the recurrent states are not interpreted as physical traffic time steps.

---

## Experiment matrix

The paper experiments are organized as follows:

| Stage | Purpose |
|---|---|
| E1 | In-domain evaluation on CIC-IDS2017, CSE-CIC-IDS2018, and CICIoT2023 |
| E2 | Bidirectional related-domain transfer: 2017 -> 2018 and 2018 -> 2017 |
| E3 | Heterogeneous transfer: 2017 -> IoT2023 and 2018 -> IoT2023 |
| E4 | Multi-source transfer: 2017 + 2018 -> IoT2023 |
| E5 | DANN gradient-reversal strength sensitivity |
| E6 | Constraint-aware FGSM/PGD robustness evaluation |
| E7 | Strong tabular controls with XGBoost and CatBoost |
| E8 | Class-prior and deployment-threshold audit |

The main experiment configuration is:

```text
configs/paper_experiments_v050.yaml
```

Main repeated experiments use random seeds:

```text
42, 43, 44
```

---

## Repository structure

```text
cross-domain-ids/
├── configs/
│   ├── clean_cross_domain.yaml
│   ├── datasets.yaml
│   ├── paper_experiments_v050.yaml
│   ├── reference_results.yaml
│   ├── features/
│   │   ├── ablation_groups.yaml
│   │   └── canonical_features.yaml
│   └── robustness/
│       └── e6_constraints_v051.yaml
│
├── configs_patch/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── notebooks/
├── paper/
├── scripts/
├── src/
│   └── ids/
├── tests/
│
├── .gitignore
├── CHANGELOG.md
├── README.md
├── REPRODUCTION_NOTES.md
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

The `data/` directory structure may be created locally, but raw and processed datasets are not intended to be committed to GitHub.

---

## Installation

Python 3.10 or 3.11 is recommended for the main TensorFlow pipeline.

### Windows / PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
python scripts/check_environment.py
```

Additional packages used by the strong tabular experiments can be installed with:

```powershell
python -m pip install -r configs_patch/requirements_e7_e8.txt
```

---

## Data preparation

### CIC-IDS2017 and CSE-CIC-IDS2018

Dataset sources are registered in:

```text
configs/datasets.yaml
```

Utilities are provided for downloading and inspecting the prepared legacy datasets:

```powershell
python scripts/download_data.py --dataset legacy
python scripts/inspect_dataset.py --all
```

### CICIoT2023

Prepare a protocol pool:

```powershell
python scripts/prepare_ciciot2023.py `
  --raw-dir data/raw/ciciot2023 `
  --output data/processed/ciciot2023_protocol_pool.csv `
  --sample-frac 0.05 `
  --task binary `
  --seed 20260819
```

Create the frozen development/final-test split:

```powershell
python scripts/freeze_ciciot2023_split.py `
  --input data/processed/ciciot2023_protocol_pool.csv `
  --dev-output data/processed/ciciot2023_dev.csv `
  --test-output data/processed/ciciot2023_final_test.csv `
  --seed 20260819
```

The final-test split must not be used for model fitting, hyperparameter selection, checkpoint selection, or deployment-threshold estimation.

---

## Protocol check

Before running final experiments:

```powershell
python scripts/check_paper_protocol.py `
  --config configs/paper_experiments_v050.yaml `
  --final
```

This checks the frozen experiment configuration and canonical mapping requirements.

---

## Running the main experiments

Inspect commands before execution:

```powershell
python scripts/run_paper.py --stage E3 --mode final --dry-run
```

### E1 - In-domain evaluation

```powershell
python scripts/run_paper.py --stage E1 --mode final
```

### E2 - Related-domain transfer

```powershell
python scripts/run_paper.py --stage E2 --mode final
```

### E3 - Enterprise-to-IoT transfer

```powershell
python scripts/run_paper.py --stage E3 --mode final
```

### E4 - Multi-source transfer

```powershell
python scripts/run_paper.py --stage E4 --mode final
```

### E5 - DANN sensitivity

```powershell
python scripts/run_paper.py --stage E5 --mode smoke
```

E5 is a development-only sensitivity experiment and is not used to alter the frozen final-test configuration.

### E6 - Constraint-aware adversarial robustness

```powershell
python scripts/run_paper.py --stage E6 --mode final
```

The robustness configuration is:

```text
configs/robustness/e6_constraints_v051.yaml
```

### E7 - XGBoost and CatBoost

```powershell
python scripts/run_e7_strong_tabular.py --scenario all
```

### E8 - Class-prior and threshold audit

```powershell
python scripts/run_e8_class_prior_audit.py
```

---

## Metrics

The main evaluation metrics are:

- Macro-F1
- Balanced Accuracy
- AUPRC
- AUROC
- False Positive Rate (FPR)
- False Negative Rate (FNR)

Accuracy is retained as a supplementary statistic.

For CICIoT2023, AUPRC must be interpreted together with target attack prevalence because the frozen final-test split is strongly attack dominated.

---

## Main empirical observations

The study reports the following overall findings:

- XGBoost provides the strongest in-domain Macro-F1 across the three evaluated datasets.
- Domain adaptation is direction dependent.
- DANN improves the 2018 -> 2017 enterprise transfer but does not provide a consistent benefit in the reverse direction.
- Heterogeneous enterprise-to-IoT transfer is substantially harder than transfer between related enterprise datasets.
- The source-only canonical MLP transfers useful signal from CSE-CIC-IDS2018 to CICIoT2023, while DANN does not consistently improve that transfer.
- Combining multiple enterprise sources does not automatically make adversarial alignment beneficial.
- Stronger domain-adversarial pressure can lead to destructive over-alignment.
- Class-prior and deployment-threshold shift are distinct from representation alignment.
- Natural domain adaptation does not imply robustness to adversarial evasion.

---

## Reproducibility

Experiment runs use versioned configuration files and structured outputs rather than manual transcription of metrics.

Main utilities include:

```text
scripts/run_paper.py
scripts/build_paper_tables.py
scripts/aggregate_results.py
scripts/check_paper_protocol.py
scripts/run_e7_strong_tabular.py
scripts/run_e8_class_prior_audit.py
scripts/plot_e5_lambda_sensitivity.py
scripts/plot_e6_robustness.py
```

Historical notebook reproduction and the clean research protocol are intentionally kept separate. See:

```text
REPRODUCTION_NOTES.md
docs/LEGACY_VS_CLEAN.md
```

---

## Related publication

This repository supports the manuscript:

> Nguyen T. Phong, Pham Thi Thuy An, Bui Huong Giang, and Tin T. Tran,  
> **Domain-Adaptive Intrusion Detection across Heterogeneous Enterprise and IoT Networks**.

The study substantially extends the earlier conference work:

> V. T. Le, N. T. Phong, and T. T. Tran,  
> **Spatio-Temporal Deep Intrusion Detection with CNN-BiLSTM-Attention and Domain-Adversarial Adaptation Network**,  
> ICCIES 2026, Communications in Computer and Information Science, Springer.  
> DOI: 10.1007/978-3-032-21628-1_8

The extension adds CICIoT2023, semantically audited heterogeneous feature harmonization, stronger boosted-tree controls, multi-source transfer, DANN-strength sensitivity, class-prior and threshold analysis, and constraint-aware adversarial robustness.

---

## Data availability

The three benchmark datasets used in this study are publicly available from their respective providers.

This repository provides code, preprocessing procedures, configuration files, canonical feature mappings, experiment scripts, and reproducibility utilities. Large raw and processed dataset files are not included in the repository.

---

## Citation

If you use this repository, please cite the related publication once bibliographic details are finalized.

---

## Contact

**Tin T. Tran**  
Artificial Intelligence Laboratory  
Faculty of Information Technology  
Ton Duc Thang University  
Ho Chi Minh City, Vietnam  

Email: trantrungtin@tdtu.edu.vn
