#!/usr/bin/env python3
"""Create the E5 lambda-sensitivity figure from F3_lambda_sensitivity_rows.csv."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found in columns {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/paper_tables/F3_lambda_sensitivity_rows.csv"))
    ap.add_argument("--output", type=Path, default=Path("paper/fig/e5_lambda_sensitivity.pdf"))
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    x = pick(df, ["lambda_d", "lambda", "grl_lambda"])
    mf = pick(df, ["macro_f1", "target_macro_f1"])
    au = pick(df, ["auroc", "target_auroc"])
    dom = next((c for c in ["final_val_domain_accuracy", "domain_accuracy", "domain_acc"] if c in df.columns), None)
    df = df.sort_values(x)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(df[x], df[mf], marker="o", label="Macro-F1")
    ax.plot(df[x], df[au], marker="s", label="AUROC")
    if dom:
        ax.plot(df[x], df[dom], marker="^", label="Domain accuracy")
    ax.set_xlabel(r"GRL strength $\lambda_d$")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(args.output)

if __name__ == "__main__":
    main()
