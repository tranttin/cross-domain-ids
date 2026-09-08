#!/usr/bin/env python3
"""Aggregate E7 XGBoost/CatBoost runs across seeds into paper-ready mean±SD CSV."""
import argparse
from pathlib import Path
import pandas as pd

METRICS = ["macro_f1", "balanced_accuracy", "auprc", "auroc", "fpr", "fnr"]


def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, default=Path("results/paper_E7_strong_tabular/metrics.csv"))
    ap.add_argument("--output", type=Path, default=Path("results/paper_tables/E7_strong_tabular_mean_std.csv"))
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    keys = ["scenario", "model", "train_domain", "test_domain", "target_threshold_strategy"]
    agg = df.groupby(keys, dropna=False)[METRICS].agg(["mean", "std"]).reset_index()
    flat = agg[keys].copy()
    for m in METRICS:
        flat[m] = [fmt(a, 0.0 if pd.isna(b) else b) for a, b in zip(agg[(m,"mean")], agg[(m,"std")])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(args.output, index=False)
    print(flat.to_string(index=False))
    print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    main()
