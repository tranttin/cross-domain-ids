# Results and Logging

## Global summary

All experiment scripts append comparable evaluation rows to:

```text
results/summary.csv
```

Typical columns:

- `run_id`, `experiment`, `protocol`, `seed`;
- `model`, `train_domain`, `test_domain`, `task`;
- `accuracy`, `precision`, `recall`, `f1`;
- `macro_f1`, `balanced_accuracy`;
- binary probability metrics `auprc`, `auroc` when scores are available;
- `fpr`, `fnr`, confusion-matrix counts;
- `train_seconds`, `best_epoch` when applicable.

## Per-run directory

Each run gets an immutable-ish directory under `results/runs/<run_id>/`. If a run is
important enough to appear in a paper, preserve the directory or archive it externally.

## Quick chart

```powershell
python scripts/plot_results.py --metric f1
python scripts/plot_results.py --metric macro_f1 --protocol clean_uda_source_labelled_target_unlabelled
```

Charts are written to `results/figures/` by default.

## Reproduction delta tables

```powershell
python scripts/compare_legacy_reference.py
```

This writes:

```text
results/reproduction_delta_table1.csv
results/reproduction_delta_table2.csv
```

These are useful for deciding whether a rerun is a faithful reproduction or a new result.

## Paper discipline

Never transcribe final numbers directly from terminal screenshots. Use a logged CSV row
identified by run_id. For multi-seed experiments, aggregate the CSV programmatically.

## Binary imbalance diagnostics (clean UDA v2)

Because the historical processed datasets were balanced per multiclass class, binary
collapse can create an attack-majority distribution. Every clean binary row now logs:

- `positive_prevalence`: attack fraction in the held-out target evaluation set;
- `predicted_positive_rate`: fraction predicted as attack;
- `auprc_baseline`: target positive prevalence (the no-skill AUPRC reference);
- `auprc_lift`: AUPRC minus prevalence;
- `fpr` and `fnr`;
- `class_weight_mode`, `source_weight_0`, `source_weight_1`;
- `decision_threshold` and the source-validation metric used to choose it;
- `lambda_d` and `domain_loss_weight` for DANN.

A model that predicts almost everything as Attack can have high Accuracy/F1 when the
target is attack-majority. For research conclusions, inspect Macro-F1, Balanced
Accuracy, AUPRC relative to its prevalence baseline, FPR, and FNR together.
