# Migrating from v0.3 to v0.4

Do not copy `results/` into a new run and do not delete old v0.3 artefacts. Keep them as
diagnostic history. Replace the code tree with v0.4, reinstall editable package, and run:

```powershell
python -m pip install -e .
python scripts/check_environment.py
```

The legacy conference commands are unchanged.

For CICIoT2023, stop using the old direct command as the first step. Start with:

```powershell
python scripts/diagnose_domain_shift.py `
  --source-csv data/processed/cleaned_data_sampled.csv `
  --source-name 2018 `
  --target-csv data/processed/ciciot2023_sample.csv `
  --target-name CICIoT2023 `
  --allow-unverified-mapping
```

Then inspect/verify the feature dictionary and run the cheap baselines. A direct long
DANN run is now the **last** step, not the first.
