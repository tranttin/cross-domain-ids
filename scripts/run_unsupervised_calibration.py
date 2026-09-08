#!/usr/bin/env python
"""Evaluate label-free target-threshold adaptation on cheap source-only baselines.

Thresholds are selected from the TARGET UNLABELLED ADAPTATION pool, never from
held-out target-test labels. Target-test labels are used once, after threshold
freezing, to report evaluation metrics. Oracle target threshold is optional
DIAGNOSTIC context only and is never a candidate strategy.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier

from ids.data.canonical_mapping import align_frames_with_mapping, mapping_dataframe, resolve_mapping
from ids.data.loaders import load_csv
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.score_shift import choose_target_oracle_threshold, positive_class_score
from ids.evaluation.thresholds import choose_binary_threshold
from ids.evaluation.unsupervised_thresholds import choose_unsupervised_target_threshold
from ids.preprocessing.cross_domain import prepare_clean_source_target
from ids.utils.seed import set_global_seed


STRATEGIES = ["source", "gmm_logit", "valley_logit", "source_prior_quantile"]


def _models(seed: int):
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "SGDClassifier": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, tol=1e-4, random_state=seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--target-csv", required=True)
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--allow-unverified-mapping", action="store_true")
    ap.add_argument("--model", choices=["LogisticRegression", "SGDClassifier", "all"], default="all")
    ap.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
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

    chosen = _models(args.seed)
    if args.model != "all":
        chosen = {args.model: chosen[args.model]}

    rows = []
    threshold_records = []
    for model_name, model in chosen.items():
        model.fit(split.X_source_train, split.y_source_train)
        val_score, _ = positive_class_score(model, split.X_source_val)
        source_t, source_val_metric = choose_binary_threshold(
            split.y_source_val, val_score, strategy="source_val_macro_f1"
        )
        adapt_score, _ = positive_class_score(model, split.X_target_unlabeled)
        test_score, _ = positive_class_score(model, split.X_target_test)
        source_prev = float(np.mean(split.y_source_train))

        # Diagnostic upper bound only; deliberately excluded from args.strategies.
        oracle = choose_target_oracle_threshold(split.y_target_test, test_score)

        for strategy in args.strategies:
            chosen_t = choose_unsupervised_target_threshold(
                strategy,
                target_adapt_score=adapt_score,
                source_threshold=source_t,
                source_positive_prevalence=source_prev,
                seed=args.seed,
            )
            pred = (test_score >= chosen_t.threshold).astype(int)
            metrics = {
                **binary(split.y_target_test, pred, test_score),
                **binary_confusion(split.y_target_test, pred),
            }
            row = {
                "model": model_name,
                "threshold_strategy": strategy,
                "threshold": chosen_t.threshold,
                "threshold_status": chosen_t.status,
                "threshold_target_labels_used": False,
                "threshold_fit_pool": "target_unlabelled_adaptation",
                "source_val_threshold": float(source_t),
                "source_val_threshold_macro_f1": float(source_val_metric),
                "source_positive_prevalence": source_prev,
                "target_adapt_positive_rate_at_threshold": chosen_t.target_adapt_positive_rate,
                "mapping_version": mapping.version,
                "feature_count": len(split.selected_feature_names),
                "diagnostic_oracle_target_threshold": oracle["threshold"],
                "diagnostic_oracle_target_macro_f1": oracle["macro_f1"],
                **metrics,
            }
            rows.append(row)
            threshold_records.append({
                "model": model_name,
                **chosen_t.to_dict(),
            })

    table = pd.DataFrame(rows)
    out_dir = Path(args.output_dir or f"results/diagnostics/unsupervised_calibration_{args.source_name}_to_{args.target_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "calibration_results.csv", index=False)
    mapping_dataframe(mapping).to_csv(out_dir / "feature_mapping.csv", index=False)
    (out_dir / "threshold_details.json").write_text(json.dumps(threshold_records, indent=2), encoding="utf-8")
    (out_dir / "PROTOCOL.json").write_text(json.dumps({
        "version": "0.4.3",
        "threshold_selection_uses_target_labels": False,
        "threshold_selection_pool": "target_unlabelled_adaptation",
        "target_test_labels_used_for": "final evaluation only; oracle fields are diagnostic upper bound only",
        "oracle_is_deployable": False,
        "mapping_verified_only": not args.allow_unverified_mapping,
    }, indent=2), encoding="utf-8")

    cols = [
        "model", "threshold_strategy", "threshold", "threshold_status",
        "target_adapt_positive_rate_at_threshold", "macro_f1", "balanced_accuracy",
        "auroc", "auprc", "fpr", "fnr", "diagnostic_oracle_target_macro_f1",
    ]
    print("\nIDS v0.4.3 LABEL-FREE TARGET CALIBRATION")
    print("=" * 64)
    print(table[cols].to_string(index=False))
    print("\nAll candidate thresholds are fitted WITHOUT target-test labels.")
    print("Oracle target Macro-F1 is DIAGNOSTIC ONLY and cannot be reported as clean UDA performance.")
    print("Saved diagnostics to:", out_dir.resolve())


if __name__ == "__main__":
    main()
