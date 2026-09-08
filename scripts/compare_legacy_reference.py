#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
import pandas as pd

from ids.utils.config import load_yaml


def compare_2018(summary: pd.DataFrame, ref: dict):
    rows = summary[(summary["test_domain"].astype(str) == "2018") & (summary["task"] == "multiclass")].copy()
    rows = rows.drop_duplicates(subset=["model"], keep="last")
    out = []
    for _, r in rows.iterrows():
        if r["model"] not in ref:
            continue
        z = {"model": r["model"]}
        for m in ["accuracy", "precision", "recall", "f1"]:
            if m in r and m in ref[r["model"]]:
                z[f"rerun_{m}"] = r[m]
                z[f"ref_{m}"] = ref[r["model"]][m]
                z[f"delta_{m}"] = r[m] - ref[r["model"]][m]
        out.append(z)
    return pd.DataFrame(out)


def compare_cross(summary: pd.DataFrame, ref: dict):
    rows = summary[summary["task"] == "binary"].copy()
    if "protocol" in rows.columns:
        rows = rows[rows["protocol"].astype(str).str.contains("legacy", case=False, na=False)]
    rows = rows.drop_duplicates(subset=["model", "train_domain", "test_domain", "protocol"], keep="last")
    out = []
    for _, r in rows.iterrows():
        model = r["model"]
        direction = f"{r['train_domain']}->{r['test_domain']}"
        if model not in ref or direction not in ref[model]:
            continue
        target = ref[model][direction]
        out.append({
            "model": model, "direction": direction,
            "rerun_accuracy": r.get("accuracy"), "ref_accuracy": target.get("accuracy"),
            "delta_accuracy": r.get("accuracy") - target.get("accuracy"),
            "rerun_f1": r.get("f1"), "ref_f1": target.get("f1"),
            "delta_f1": r.get("f1") - target.get("f1"),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--reference", default="configs/reference_results.yaml")
    args = ap.parse_args()
    summary = pd.read_csv(args.summary)
    ref = load_yaml(args.reference)
    print("\n=== Legacy 2018 multiclass ===")
    t1 = compare_2018(summary, ref["legacy_2018_multiclass"])
    print(t1.to_string(index=False) if len(t1) else "No matching rows yet.")
    print("\n=== Legacy cross-domain ===")
    t2 = compare_cross(summary, ref["legacy_cross_domain"])
    print(t2.to_string(index=False) if len(t2) else "No matching rows yet.")
    Path("results").mkdir(exist_ok=True)
    if len(t1): t1.to_csv("results/reproduction_delta_table1.csv", index=False)
    if len(t2): t2.to_csv("results/reproduction_delta_table2.csv", index=False)


if __name__ == "__main__":
    main()
