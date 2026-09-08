# Score-shift diagnostics (v0.4.2)

A cross-domain classifier can preserve **ranking** while failing as a hard classifier.
For example, target AUROC can be high even when almost no target score crosses a
threshold chosen on labelled source validation. This is a calibration / score-location
shift, not necessarily a representation failure.

v0.4.2 therefore records three separate layers:

1. ranking: AUROC/AUPRC;
2. score distribution: source/target quantiles and class-conditional medians;
3. decision transfer: fraction of target scores above the source-selected threshold,
   Macro-F1, Balanced Accuracy, FPR/FNR.

## Dedicated diagnostic

```powershell
python scripts/diagnose_score_shift.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --model all
```

Outputs include per-model score quantiles and `score_shift_summary.csv`.

The script also computes a **target-labelled oracle threshold** to answer a narrow
scientific question: *if the representation ranks target samples correctly, how much
performance is lost only because the source threshold does not transfer?*

The oracle is strictly diagnostic. It must never be used for UDA training, checkpoint
selection, final threshold selection, or reported as a deployable clean-UDA result.

## RF orientation check

Probability columns are resolved through `model.classes_`; code never assumes that
`predict_proba(... )[:, 1]` is automatically class 1. v0.4.2 also reports target AUROC
for `score` and `1-score`. If raw AUROC <= 0.25 while reversed AUROC >= 0.75, the run is
flagged `diagnostic_orientation_suspect`. Do not automatically reverse a model for the
paper: first review feature semantics and domain-specific decision behavior.

## Mapping ablation

```powershell
python scripts/run_mapping_ablation.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping `
  --mode groups `
  --models LogisticRegression RandomForest
```

Predefined groups live in `configs/features/ablation_groups.yaml` and include the full
candidate set, a set without RST/URG counts, a rates/lengths core, no-flags, and
flags-only diagnostics. `--mode loo` performs leave-one-feature-out diagnostics.

**Do not choose the final mapping by whichever target-test row scores highest.** The
ablation is used to identify features that deserve semantic/unit review. Final feature
inclusion must be justified independently from target-test labels.
