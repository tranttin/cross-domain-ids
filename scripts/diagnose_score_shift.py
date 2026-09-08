#!/usr/bin/env python
"""Diagnose cross-domain score/calibration shift without changing clean-UDA rules."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from ids.data.canonical_mapping import align_frames_with_mapping, mapping_dataframe, resolve_mapping
from ids.data.loaders import load_csv
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.score_shift import (
    build_score_shift_diagnostic,
    positive_class_score,
    score_quantile_frame,
)
from ids.evaluation.thresholds import choose_binary_threshold
from ids.preprocessing.cross_domain import binary_labels_for_dataset, prepare_clean_source_target
from ids.utils.seed import set_global_seed


def _stratified_cap(df: pd.DataFrame, name: str, n: int | None, seed: int) -> pd.DataFrame:
    if not n or n >= len(df):
        return df
    y = binary_labels_for_dataset(df, name)
    rng = np.random.default_rng(seed)
    parts = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        take = max(1, int(round(n * len(idx) / len(df))))
        chosen = rng.choice(idx, size=min(take, len(idx)), replace=False)
        parts.append(df.iloc[chosen])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _models(seed: int):
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=seed),
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
    ap.add_argument("--model", choices=["LogisticRegression", "RandomForest", "SGDClassifier", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-source", type=int, default=100000)
    ap.add_argument("--max-target", type=int, default=50000)
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    set_global_seed(args.seed)

    source = _stratified_cap(load_csv(args.source_csv), args.source_name, args.max_source, args.seed)
    target = _stratified_cap(load_csv(args.target_csv), args.target_name, args.max_target, args.seed + 1)
    mapping = resolve_mapping(
        source.columns, target.columns, args.source_name, args.target_name,
        mapping_path=args.mapping, verified_only=not args.allow_unverified_mapping,
    )
    if not mapping.features:
        raise SystemExit("No usable mapping. Run diagnose_domain_shift.py first.")
    source, target = align_frames_with_mapping(source, target, mapping)
    split = prepare_clean_source_target(
        source, target, args.source_name, args.target_name,
        seed=args.seed, threshold=0.0, feature_alignment="exact",
    )

    out_dir = Path(args.output_dir or f"results/diagnostics/score_shift_{args.source_name}_to_{args.target_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_dataframe(mapping).to_csv(out_dir / "feature_mapping.csv", index=False)
    (out_dir / "selected_features.json").write_text(
        json.dumps(split.selected_feature_names, indent=2), encoding="utf-8"
    )

    chosen = _models(args.seed)
    if args.model != "all":
        chosen = {args.model: chosen[args.model]}

    summaries = []
    for name, model in chosen.items():
        t0 = time.time()
        model.fit(split.X_source_train, split.y_source_train)
        val_score, meta = positive_class_score(model, split.X_source_val)
        threshold, source_val_metric = choose_binary_threshold(
            split.y_source_val, val_score, strategy="source_val_macro_f1"
        )
        target_score, target_meta = positive_class_score(model, split.X_target_test)
        pred = (target_score >= threshold).astype(int)
        task = {**binary(split.y_target_test, pred, target_score), **binary_confusion(split.y_target_test, pred)}
        diag = build_score_shift_diagnostic(
            source_val_score=val_score,
            source_val_labels=split.y_source_val,
            source_threshold=threshold,
            target_score=target_score,
            target_labels=split.y_target_test,  # diagnostic only; never selection/training
        )
        row = {
            "model": name,
            "mapping_version": mapping.version,
            "feature_count": len(split.selected_feature_names),
            "score_method": meta.method,
            "model_classes": meta.classes,
            "positive_class_index": meta.positive_class_index,
            "target_score_method": target_meta.method,
            "source_threshold": threshold,
            "source_threshold_metric": source_val_metric,
            "train_seconds": round(time.time() - t0, 3),
            **task,
            **diag,
        }
        summaries.append(row)

        slug = name.lower().replace("classifier", "").replace("regression", "").strip("_")
        score_quantile_frame(
            val_score, target_score,
            source_val_labels=split.y_source_val,
            target_labels=split.y_target_test,
        ).to_csv(out_dir / f"{slug}_score_quantiles.csv", index=False)
        (out_dir / f"{slug}_score_shift.json").write_text(
            json.dumps(row, indent=2, default=str), encoding="utf-8"
        )

    table = pd.DataFrame(summaries)
    table.to_csv(out_dir / "score_shift_summary.csv", index=False)
    cols = [
        "model", "auroc", "macro_f1", "balanced_accuracy", "source_threshold",
        "target_fraction_above_source_threshold", "target_score_median",
        "diagnostic_target_benign_score_median", "diagnostic_target_attack_score_median",
        "diagnostic_target_oracle_threshold", "diagnostic_target_oracle_macro_f1",
        "diagnostic_target_auroc_reversed", "diagnostic_orientation_suspect",
    ]
    print("\nIDS v0.4.2 SCORE-SHIFT DIAGNOSTIC")
    print("=" * 56)
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    print("\nIMPORTANT: target-labelled class quantiles and oracle threshold are DIAGNOSTIC ONLY.")
    print("They must not be used for UDA training, checkpoint selection, or final deployable thresholding.")
    print("Saved diagnostics to:", out_dir.resolve())


if __name__ == "__main__":
    main()
