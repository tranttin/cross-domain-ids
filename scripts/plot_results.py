#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--metric", default="f1")
    ap.add_argument("--experiment")
    ap.add_argument("--protocol")
    ap.add_argument("--task")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    for col, value in [("experiment", args.experiment), ("protocol", args.protocol), ("task", args.task)]:
        if value and col in df.columns:
            df = df[df[col] == value]
    if args.metric not in df.columns:
        raise SystemExit(f"Metric '{args.metric}' not found. Available: {', '.join(df.columns)}")
    df = df.dropna(subset=[args.metric]).copy()
    if df.empty:
        raise SystemExit("No rows after filtering.")

    def label(row):
        direction = ""
        if "train_domain" in row and "test_domain" in row:
            direction = f" {row.get('train_domain','')}→{row.get('test_domain','')}"
        hp = ""
        if row.get("model") == "Clean-DANN" and "lambda_d" in row and pd.notna(row.get("lambda_d")):
            hp += f" λ={float(row.get('lambda_d')):g}"
        if "class_weight_mode" in row and pd.notna(row.get("class_weight_mode")):
            hp += f" cw={row.get('class_weight_mode')}"
        return f"{row.get('model','model')}{direction}{hp}"

    df["label"] = df.apply(label, axis=1)
    # Keep the latest occurrence of each comparable row for a clean quick-look chart.
    df = df.drop_duplicates(subset=["label"], keep="last").sort_values(args.metric)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(df) + 1.5)))
    ax.barh(df["label"], df[args.metric])
    ax.set_xlabel(args.metric)
    ax.set_title(f"IDS experiment comparison — {args.metric}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    output = Path(args.output or f"results/figures/{args.metric}_comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print("Saved:", output)


if __name__ == "__main__":
    main()
