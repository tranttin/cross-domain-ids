#!/usr/bin/env python3
"""Generate one robustness curve per attack from final E6 row-level CSV."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("results/paper_tables/E6_robustness_rows.csv"))
    ap.add_argument("--output-dir", type=Path, default=Path("paper/fig"))
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    required = {"attack", "epsilon", "model", "robust_macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for attack, sub in df.groupby("attack"):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for model, g in sub.groupby("model"):
            g = g.sort_values("epsilon")
            ax.plot(g["epsilon"], g["robust_macro_f1"], marker="o", label=model)
        ax.set_xlabel(r"Perturbation budget $\epsilon$")
        ax.set_ylabel("Robust Macro-F1")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        stem = f"e6_robustness_{str(attack).lower()}"
        fig.savefig(args.output_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(args.output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(args.output_dir / f"{stem}.pdf")

if __name__ == "__main__":
    main()
