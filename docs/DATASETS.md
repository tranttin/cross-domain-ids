# Dataset Registry and Expansion Plan

Dataset metadata and automatic download information live in `configs/datasets.yaml`.
The code distinguishes **official scientific source** from **convenient download mirror**.
Always cite the original dataset publication/source in research papers.

## Legacy reproduction datasets

### CSE-CIC-IDS2018 processed

Automatic Kaggle input used by the recovered notebook:

```text
hunglevinh/cse-cic-ids2018-cleaned-40k-samples-each
```

Expected processed file: `cleaned_data_sampled.csv`, shape `(360000, 79)` in the notebook.

### CIC-IDS2017 balanced 77 features

Recovered exact Kaggle slug:

```text
ltrngvinh/cicids2017-balanced-77feats
```

Expected file: `cicids2017_balanced_77feats.csv`, shape `(137992, 78)`.

These two custom processed files are required for historical reproduction.

## Recommended third domain: CICIoT2023

Official source:

```text
https://www.unb.ca/cic/datasets/iotdataset-2023.html
```

The official page describes 33 attacks grouped into DDoS, DoS, Recon, Web-based,
Brute Force, Spoofing and Mirai, collected in an IoT topology with 105 devices.
It provides PCAP and CSV data.

For convenience the registry includes a public Kaggle mirror. It is opt-in because the
corpus is large:

```powershell
python scripts/download_data.py --dataset ciciot2023
python scripts/prepare_ciciot2023.py --sample-frac 0.01 --task binary
```

The preparation script samples in chunks instead of concatenating the full corpus in RAM.
For final experiments, freeze a documented sampling strategy rather than changing
`--sample-frac` opportunistically.

## Diverse non-CIC benchmark: UNSW-NB15

Official source:

```text
https://research.unsw.edu.au/projects/unsw-nb15-dataset
```

The official site reports 2,540,044 total records, 49 features and nine attack families,
and provides author-defined training (175,341) and testing (82,332) subsets.

Convenience flow:

```powershell
python scripts/download_data.py --dataset unsw_nb15
python scripts/prepare_unsw_nb15.py --task binary
```

The adapter preserves the provided train/test split and writes normalized CSV files.

## IoT malware domain: IoT-23

Official source:

```text
https://www.stratosphereips.org/datasets-iot23
```

The official site describes 23 scenarios: 20 malware captures and 3 benign IoT captures.
The full archive and a flow-only archive are both large, so automatic downloading is
intentionally disabled. Download manually, document the exact version, and add a parser
only after deciding whether experiments use original Zeek flow logs or a third-party CSV
conversion.

## Recommended dataset progression

For the thesis/journal line:

```text
Stage A: CIC-IDS2017 <-> CSE-CIC-IDS2018
         reproduce legacy + establish clean DANN baseline

Stage B: CICIoT2023
         third-domain IoT transfer / unseen-domain generalization

Stage C: UNSW-NB15
         different data-generation family; tests whether gains are CIC-specific

Optional: IoT-23
         malware-focused IoT traffic with substantially different representation
```

Do not claim one universal cross-domain experiment unless feature semantics are actually
aligned. A smaller, defensible common feature space is preferable to arbitrary column
renaming.

## v0.4 note on CICIoT2023

The prepared CICIoT2023 CSV is intentionally kept in its native numeric feature
space. Cross-dataset semantic mapping is declared separately in
`configs/features/canonical_features.yaml`. Do not rename native columns in the
preparer merely to make them look like CICFlowMeter columns.

Candidate mappings must be audited for definition and unit before final experiments.
See `docs/FEATURE_HARMONIZATION.md`.
