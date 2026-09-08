#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


KNOWN = {
    "cse_cic_ids2018_processed": Path("data/processed/cleaned_data_sampled.csv"),
    "cic_ids2017_processed": Path("data/processed/cicids2017_balanced_77feats.csv"),
    "ciciot2023_sample": Path("data/processed/ciciot2023_sample.csv"),
    "unsw_nb15_train": Path("data/processed/unsw_nb15/unsw_nb15_train.csv"),
    "unsw_nb15_test": Path("data/processed/unsw_nb15/unsw_nb15_test.csv"),
}


def inspect(name, path):
    if not path.exists():
        print(f"[MISSING] {name}: {path}")
        return
    df = pd.read_csv(path, low_memory=False)
    print(f"\n[{name}] {path}")
    print("shape:", df.shape)
    print("columns:", len(df.columns))
    if "Label" in df.columns:
        print("Label counts:")
        print(df["Label"].value_counts(dropna=False).head(40).to_string())
    print("NaN cells:", int(df.isna().sum().sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=KNOWN)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = list(KNOWN) if args.all or not args.dataset else [args.dataset]
    for n in names:
        inspect(n, KNOWN[n])


if __name__ == "__main__":
    main()
