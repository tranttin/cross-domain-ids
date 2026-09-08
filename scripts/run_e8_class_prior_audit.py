#!/usr/bin/env python3
"""E8 label-shift / operating-point audit.

This is an audit, not a new model. It quantifies:
  1) benign/attack prevalence in each frozen dataset;
  2) target predicted-positive-rate versus true target attack prevalence from
     existing metrics.csv files;
  3) source-validation threshold versus the label-free deployment threshold;
  4) when E7 score bundles exist, performance under both thresholds without
     retraining or target-label-based threshold selection.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


def norm_name(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def find_label(df):
    m = {norm_name(c): c for c in df.columns}
    for k in ["label_bin", "label", "class", "target", "attack"]:
        if k in m:
            return m[k]
    for c in df.columns:
        if "label" in norm_name(c):
            return c
    raise ValueError("No label column found")


def binarize(s):
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").fillna(0).to_numpy()
        if set(np.unique(v)).issubset({0, 1, 0.0, 1.0}):
            return v.astype(np.int8)
        return (v != 0).astype(np.int8)
    t = s.astype(str).str.upper().str.strip()
    return (~t.str.contains(r"BENIGN|NORMAL", regex=True, na=False)).astype(np.int8).to_numpy()


def prevalence_row(name: str, path: Path):
    df = pd.read_csv(path, low_memory=False)
    y = binarize(df[find_label(df)])
    return {
        "dataset": name,
        "n": len(y),
        "benign_n": int(np.sum(y == 0)),
        "attack_n": int(np.sum(y == 1)),
        "benign_rate": float(np.mean(y == 0)),
        "attack_rate": float(np.mean(y == 1)),
    }


def threshold_metrics(y, score, threshold):
    pred = (score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "fpr": fp / max(tn + fp, 1),
        "fnr": fn / max(tp + fn, 1),
        "predicted_positive_rate": float(np.mean(pred)),
    }


def read_existing_metrics(results_root: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in results_root.rglob("metrics.csv"):
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if "test_domain" not in d.columns:
            continue
        d = d.copy()
        d["metrics_file"] = str(p)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-2017", type=Path, default=Path("data/processed/cicids2017_balanced_77feats.csv"))
    ap.add_argument("--data-2018", type=Path, default=Path("data/processed/cleaned_data_sampled.csv"))
    ap.add_argument("--iot-test", type=Path, default=Path("data/processed/ciciot2023_final_test.csv"))
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--e7-scores", type=Path, default=Path("results/paper_E7_strong_tabular"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/paper_E8_label_shift"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    priors = pd.DataFrame([
        prevalence_row("CIC-IDS2017", args.data_2017),
        prevalence_row("CSE-CIC-IDS2018", args.data_2018),
        prevalence_row("CICIoT2023_FINAL_TEST", args.iot_test),
    ])
    priors.to_csv(args.output_dir / "E8_dataset_class_priors.csv", index=False)
    print("\n[E8] Class priors")
    print(priors.to_string(index=False))

    metrics = read_existing_metrics(args.results_root)
    if not metrics.empty:
        iot = metrics[metrics["test_domain"].astype(str).str.contains("IoT", case=False, na=False)].copy()
        needed = [
            "paper_stage", "experiment", "seed", "model", "train_domain", "test_domain",
            "positive_prevalence", "predicted_positive_rate", "source_val_decision_threshold",
            "decision_threshold", "target_threshold_strategy", "macro_f1", "balanced_accuracy",
            "auprc", "auroc", "fpr", "fnr", "metrics_file"
        ]
        for c in needed:
            if c not in iot.columns:
                iot[c] = np.nan
        iot["prior_prediction_gap"] = iot["predicted_positive_rate"] - iot["positive_prevalence"]
        iot["threshold_shift"] = iot["decision_threshold"] - iot["source_val_decision_threshold"]
        iot[needed + ["prior_prediction_gap", "threshold_shift"]].to_csv(
            args.output_dir / "E8_existing_iot_threshold_audit_rows.csv", index=False
        )

        group_keys = ["model", "train_domain", "test_domain", "target_threshold_strategy"]
        cols = ["positive_prevalence", "predicted_positive_rate", "prior_prediction_gap", "source_val_decision_threshold", "decision_threshold", "threshold_shift", "macro_f1", "fpr", "fnr"]
        summary = iot.groupby(group_keys, dropna=False)[cols].agg(["mean", "std", "count"])
        summary.to_csv(args.output_dir / "E8_existing_iot_threshold_audit_summary.csv")
        print("\n[E8] Existing IoT metrics audit saved.")

    # E7 score bundles permit a clean source-threshold vs deployment-threshold comparison.
    rows = []
    for p in args.e7_scores.glob("scores_*_s*.npz"):
        z = np.load(p)
        y = z["y_test"].astype(np.int8)
        score = z["test_scores"].astype(float)
        st = float(z["source_val_threshold"][0])
        dt = float(z["decision_threshold"][0])
        a = threshold_metrics(y, score, st)
        b = threshold_metrics(y, score, dt)
        rows.append({
            "score_bundle": p.name,
            "target_attack_rate": float(np.mean(y)),
            "source_threshold": st,
            "deployment_threshold": dt,
            **{f"source_{k}": v for k, v in a.items()},
            **{f"deploy_{k}": v for k, v in b.items()},
            "delta_macro_f1": b["macro_f1"] - a["macro_f1"],
            "delta_fpr": b["fpr"] - a["fpr"],
            "delta_fnr": b["fnr"] - a["fnr"],
        })
    if rows:
        pd.DataFrame(rows).to_csv(args.output_dir / "E8_E7_dual_threshold_metrics.csv", index=False)
        print("[E8] Dual-threshold E7 audit saved.")

    print(f"\n[E8] Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
