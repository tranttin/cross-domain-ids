#!/usr/bin/env python
"""Audit protocol encodings before deriving cross-dataset one-hot protocol features."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--target-csv")
    ap.add_argument("--source-column", default="Protocol")
    args = ap.parse_args()

    src = pd.read_csv(args.source_csv, usecols=[args.source_column])
    print("SOURCE PROTOCOL VALUES")
    counts = src[args.source_column].value_counts(dropna=False).sort_index()
    print(counts.to_string())
    print("\nKnown IANA values commonly relevant here: ICMP=1, TCP=6, UDP=17.")
    print("Do not derive protocol one-hot features unless the observed encoding is confirmed.")

    if args.target_csv:
        tgt = pd.read_csv(args.target_csv, usecols=lambda c: c in {"Protocol Type", "TCP", "UDP", "ICMP"})
        print("\nTARGET PROTOCOL COLUMNS")
        for c in tgt.columns:
            print(f"\n{c}")
            print(tgt[c].value_counts(dropna=False).head(20).to_string())


if __name__ == "__main__":
    main()
