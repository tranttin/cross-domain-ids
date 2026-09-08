#!/usr/bin/env python
"""Source-led feature stability audit for canonical cross-domain features.

This script does NOT select features using target labels. It trains a balanced
Logistic Regression on labelled source train data, reports source-validation
coefficient/ablation evidence, and combines it with UNLABELLED target-adaptation
feature-distribution shift. Target-test labels are never read for selection.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from ids.data.canonical_mapping import align_frames_with_mapping, mapping_dataframe, resolve_mapping
from ids.data.loaders import load_csv
from ids.evaluation.score_shift import positive_class_score
from ids.evaluation.thresholds import choose_binary_threshold
from ids.preprocessing.cross_domain import prepare_clean_source_target
from ids.utils.seed import set_global_seed


def _safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def _feature_stats(x_source, x_target):
    s = np.asarray(x_source, dtype=float)
    t = np.asarray(x_target, dtype=float)
    sp01, smed, sp99 = np.quantile(s, [0.01, 0.50, 0.99])
    tp01, tmed, tp99 = np.quantile(t, [0.01, 0.50, 0.99])
    denom = max(float(sp99 - sp01), 1e-9)
    return {
        "source_scaled_p01": float(sp01),
        "source_scaled_median": float(smed),
        "source_scaled_p99": float(sp99),
        "target_adapt_scaled_p01": float(tp01),
        "target_adapt_scaled_median": float(tmed),
        "target_adapt_scaled_p99": float(tp99),
        "median_shift_in_source_p01_p99_units": float((tmed - smed) / denom),
        "target_outside_source_p01_p99_fraction": float(np.mean((t < sp01) | (t > sp99))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--target-csv", required=True)
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--allow-unverified-mapping", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    set_global_seed(args.seed)

    source = load_csv(args.source_csv)
    target = load_csv(args.target_csv)
    mapping = resolve_mapping(
        source.columns, target.columns, args.source_name, args.target_name,
        mapping_path=args.mapping, verified_only=not args.allow_unverified_mapping,
    )
    if not mapping.features:
        raise SystemExit("No usable mapping. Run diagnose_domain_shift.py first.")
    source, target = align_frames_with_mapping(source, target, mapping)
    split = prepare_clean_source_target(
        source, target, args.source_name, args.target_name,
        seed=args.seed, threshold=0.01, feature_alignment="exact",
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
    model.fit(split.X_source_train, split.y_source_train)
    val_score, _ = positive_class_score(model, split.X_source_val)
    source_threshold, source_metric = choose_binary_threshold(
        split.y_source_val, val_score, strategy="source_val_macro_f1"
    )
    val_pred = (val_score >= source_threshold).astype(int)
    full_macro = float(f1_score(split.y_source_val, val_pred, average="macro", zero_division=0))
    full_bal = float(balanced_accuracy_score(split.y_source_val, val_pred))
    full_auc = _safe_auc(split.y_source_val, val_score)

    coefs = np.asarray(model.coef_).reshape(-1)
    rows = []
    for j, name in enumerate(split.selected_feature_names):
        stats = _feature_stats(split.X_source_val[:, j], split.X_target_unlabeled[:, j])
        source_contrib = coefs[j] * split.X_source_val[:, j]
        target_contrib = coefs[j] * split.X_target_unlabeled[:, j]

        keep = np.ones(len(split.selected_feature_names), dtype=bool)
        keep[j] = False
        if keep.sum() >= 1:
            loo = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
            loo.fit(split.X_source_train[:, keep], split.y_source_train)
            loo_score, _ = positive_class_score(loo, split.X_source_val[:, keep])
            loo_t, _ = choose_binary_threshold(split.y_source_val, loo_score, strategy="source_val_macro_f1")
            loo_pred = (loo_score >= loo_t).astype(int)
            loo_macro = float(f1_score(split.y_source_val, loo_pred, average="macro", zero_division=0))
            loo_bal = float(balanced_accuracy_score(split.y_source_val, loo_pred))
            loo_auc = _safe_auc(split.y_source_val, loo_score)
        else:
            loo_macro = loo_bal = loo_auc = float("nan")

        rows.append({
            "feature": name,
            "lr_coefficient": float(coefs[j]),
            "abs_lr_coefficient": float(abs(coefs[j])),
            "source_val_full_macro_f1": full_macro,
            "source_val_full_balanced_accuracy": full_bal,
            "source_val_full_auroc": full_auc,
            "source_val_loo_macro_f1": loo_macro,
            "source_val_loo_balanced_accuracy": loo_bal,
            "source_val_loo_auroc": loo_auc,
            "source_val_macro_f1_drop_when_removed": float(full_macro - loo_macro),
            "source_val_auroc_drop_when_removed": float(full_auc - loo_auc),
            "source_abs_contribution_median": float(np.median(np.abs(source_contrib))),
            "target_adapt_abs_contribution_median": float(np.median(np.abs(target_contrib))),
            "target_to_source_abs_contribution_ratio": float(
                np.median(np.abs(target_contrib)) / max(np.median(np.abs(source_contrib)), 1e-12)
            ),
            **stats,
        })

    table = pd.DataFrame(rows)
    table["coefficient_abs_rank"] = table["abs_lr_coefficient"].rank(method="dense", ascending=False).astype(int)
    table = table.sort_values(
        ["source_val_auroc_drop_when_removed", "abs_lr_coefficient"], ascending=[False, False]
    ).reset_index(drop=True)

    out_dir = Path(args.output_dir or f"results/diagnostics/feature_stability_{args.source_name}_to_{args.target_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "feature_stability.csv", index=False)
    mapping_dataframe(mapping).to_csv(out_dir / "feature_mapping.csv", index=False)
    (out_dir / "source_only_summary.json").write_text(json.dumps({
        "mapping_version": mapping.version,
        "mapping_verified_only": not args.allow_unverified_mapping,
        "selected_feature_count": len(split.selected_feature_names),
        "source_threshold": float(source_threshold),
        "source_threshold_macro_f1": float(source_metric),
        "source_val_macro_f1": full_macro,
        "source_val_balanced_accuracy": full_bal,
        "source_val_auroc": full_auc,
        "target_labels_used_for_selection": False,
        "target_distribution_used": "unlabelled adaptation pool only",
    }, indent=2), encoding="utf-8")

    cols = [
        "feature", "lr_coefficient", "source_val_auroc_drop_when_removed",
        "source_val_macro_f1_drop_when_removed", "median_shift_in_source_p01_p99_units",
        "target_outside_source_p01_p99_fraction", "target_to_source_abs_contribution_ratio",
    ]
    print("\nIDS v0.4.3 FEATURE-STABILITY AUDIT")
    print("=" * 60)
    print(table[cols].to_string(index=False))
    print("\nSelection evidence above is SOURCE-validation + UNLABELLED target-adaptation shift only.")
    print("Target-test labels are not used to choose or rank features in this audit.")
    print("Saved diagnostics to:", out_dir.resolve())


if __name__ == "__main__":
    main()
