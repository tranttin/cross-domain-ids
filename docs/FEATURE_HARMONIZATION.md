# Feature harmonization protocol (v0.4)

Cross-dataset IDS experiments are only meaningful when a shared feature name also has
a shared **semantic definition**. Formatting equality is not enough.

## Policy

`configs/features/canonical_features.yaml` is the only place where semantic mappings
are declared. A feature is admitted to a final experiment only when both dataset
entries have `verified: true`.

CICIoT2023 entries ship as **candidates**, not claims. The student must check the
feature definition/unit in the official dataset documentation or extraction code, add a
short note, and then change `verified: false` to `verified: true` only when justified.

Never solve a low-overlap problem by arbitrary renaming.

## Required workflow

1. Prepare the native dataset without changing feature meaning.
2. Run formatting-only overlap for orientation.
3. Run the explicit mapping diagnostic with `--allow-unverified-mapping`.
4. Inspect `feature_stats.csv`, `out_of_range.csv`, and `scaled_shift.csv`.
5. Verify candidate mappings one by one.
6. Run LR/RF/SGD transfer sanity checks.
7. Only then run the deep backbone or DANN.

Example diagnostic:

```powershell
python scripts/diagnose_domain_shift.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping
```

The target labels read by this script are for **offline diagnostics only**. They are not
passed into UDA training or threshold selection.

## Candidate versus final runs

A command using `--allow-unverified-mapping` is automatically tagged as diagnostic.
Do not put its number in the final manuscript.

Final runs must omit that flag. If fewer than the required number of verified features
remain, the runner stops rather than silently guessing.


## v0.4.1 correction

For CICIoT2023, `flow_duration` is the flow-duration candidate. The separate `Duration` feature is not treated as flow duration. Flag counts use the `*_count` fields; `*_flag_number` fields are not substituted for counts. Raw protocol integers are excluded until audited.
