#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--metric", action="append", default=[])
    ap.add_argument("--protocol")
    ap.add_argument("--output", default="results/aggregate_mean_std.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    if args.protocol and "protocol" in df.columns:
        df = df[df["protocol"] == args.protocol]
    metrics = args.metric or [m for m in ["accuracy", "f1", "macro_f1", "balanced_accuracy", "auprc", "auroc", "fpr", "fnr"] if m in df.columns]
    groups = [c for c in [
        "protocol", "model", "train_domain", "test_domain", "task",
        "class_weight_mode", "threshold_strategy", "lambda_d", "domain_loss_weight",
    ] if c in df.columns]
    agg = df.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([x for x in col if x]) if isinstance(col, tuple) else col for col in agg.columns]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False)
    print(agg.to_string(index=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
