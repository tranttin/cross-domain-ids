#!/usr/bin/env python
"""Cheap cross-domain sanity check before spending hours on CNN/DANN."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from ids.data.loaders import load_csv
from ids.data.canonical_mapping import resolve_mapping, align_frames_with_mapping, mapping_dataframe
from ids.preprocessing.cross_domain import prepare_clean_source_target, prepare_clean_source_target_fixed_test, binary_labels_for_dataset
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.thresholds import choose_binary_threshold
from ids.evaluation.score_shift import (
    build_score_shift_diagnostic,
    positive_class_score,
    score_quantile_frame,
)
from ids.experiments.tracker import ExperimentTracker
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




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--target-csv", required=True, help="Target unlabeled adaptation/dev CSV.")
    ap.add_argument("--target-test-csv", help="Optional externally frozen target test CSV.")
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--allow-unverified-mapping", action="store_true", help="Diagnostic only; final experiments require verified mappings.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-source", type=int)
    ap.add_argument("--max-target", type=int)
    ap.add_argument("--output-root", default="results")
    args = ap.parse_args()
    set_global_seed(args.seed)

    source = _stratified_cap(load_csv(args.source_csv), args.source_name, args.max_source, args.seed)
    target = _stratified_cap(load_csv(args.target_csv), args.target_name, args.max_target, args.seed + 1)
    target_test_raw = load_csv(args.target_test_csv) if args.target_test_csv else None
    mapping = resolve_mapping(
        source.columns, target.columns, args.source_name, args.target_name,
        mapping_path=args.mapping, verified_only=not args.allow_unverified_mapping,
    )
    if not mapping.features:
        raise SystemExit("No usable feature mapping. Run scripts/diagnose_domain_shift.py first.")
    source, target = align_frames_with_mapping(source, target, mapping)
    if target_test_raw is not None:
        _source_for_test, target_test = align_frames_with_mapping(
            _stratified_cap(load_csv(args.source_csv), args.source_name, args.max_source, args.seed),
            target_test_raw, mapping,
        )
        split = prepare_clean_source_target_fixed_test(
            source, target, target_test, args.source_name, args.target_name,
            seed=args.seed, threshold=0.0, feature_alignment="exact",
        )
    else:
        split = prepare_clean_source_target(
            source, target, args.source_name, args.target_name,
            seed=args.seed, threshold=0.0, feature_alignment="exact",
        )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=args.seed),
        "SGDClassifier": SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, tol=1e-4, random_state=args.seed),
    }

    cfg = vars(args).copy()
    with ExperimentTracker(
        experiment="transfer_sanity_baselines", protocol="v0.5.0_source_only_transfer_baselines",
        seed=args.seed, config=cfg, output_root=args.output_root,
    ) as run:
        mapping_dataframe(mapping).to_csv(run.run_dir / "feature_mapping.csv", index=False)
        run.save_feature_names(split.selected_feature_names, "selected_features.json")
        rows = []
        for name, model in models.items():
            t0 = time.time()
            model.fit(split.X_source_train, split.y_source_train)
            val_score, score_meta = positive_class_score(model, split.X_source_val)
            threshold, val_metric = choose_binary_threshold(
                split.y_source_val, val_score, strategy="source_val_macro_f1"
            )
            score, target_score_meta = positive_class_score(model, split.X_target_test)
            pred = (score >= threshold).astype(int)
            m = {**binary(split.y_target_test, pred, score), **binary_confusion(split.y_target_test, pred)}
            diag = build_score_shift_diagnostic(
                source_val_score=val_score,
                source_val_labels=split.y_source_val,
                source_threshold=threshold,
                target_score=score,
                target_labels=split.y_target_test,  # diagnostic/oracle fields only
            )
            row = {
                "model": name, "train_domain": args.source_name, "test_domain": args.target_name,
                "task": "binary", "adaptation": "none", "target_labels_in_training": False,
                "mapping_version": mapping.version, "mapping_verified_only": not args.allow_unverified_mapping,
                "feature_count": len(split.selected_feature_names), "decision_threshold": threshold,
                "threshold_source_val_metric": val_metric,
                "score_method": score_meta.method, "model_classes": score_meta.classes,
                "positive_class_index": score_meta.positive_class_index,
                "target_score_method": target_score_meta.method,
                "train_seconds": round(time.time() - t0, 3), **m, **diag,
            }
            rows.append(row); run.record_metrics(row)
            score_dir = run.run_dir / "score_shift"
            score_dir.mkdir(parents=True, exist_ok=True)
            slug = name.lower().replace("classifier", "").replace("regression", "").strip("_")
            score_quantile_frame(
                val_score, score, source_val_labels=split.y_source_val, target_labels=split.y_target_test
            ).to_csv(score_dir / f"{slug}_score_quantiles.csv", index=False)
            run.write_json(f"score_shift/{slug}_summary.json", diag)

        table = pd.DataFrame(rows)
        run.save_dataframe("table.csv", table)
        all_bad = bool(((table["auroc"] <= 0.55) & (table["balanced_accuracy"] <= 0.55)).all())
        any_good = bool(((table["auroc"] >= 0.60) & (table["balanced_accuracy"] >= 0.55)).any())
        score_shift_suspect = bool((
            (table["auroc"] >= 0.75)
            & (table["balanced_accuracy"] <= 0.55)
            & ((table["target_fraction_above_source_threshold"] <= 0.05)
               | (table["target_fraction_above_source_threshold"] >= 0.95))
        ).any())
        orientation_suspect = bool(table["diagnostic_orientation_suspect"].fillna(False).any())
        if score_shift_suspect:
            status = "SCORE-SHIFT-SUSPECT"
        elif any_good:
            status = "PASS"
        elif all_bad:
            status = "DATA-MAPPING-SUSPECT"
        else:
            status = "MIXED-REVIEW"
        gate = {
            "status": status,
            "heuristic_only": True,
            "rule": (
                "SCORE-SHIFT-SUSPECT if a baseline has AUROC>=0.75 but balanced_accuracy<=0.55 "
                "and <=5% or >=95% target scores cross the SOURCE threshold; PASS if >=1 baseline "
                "has AUROC>=0.60 and balanced_accuracy>=0.55; DATA-MAPPING-SUSPECT if all are <=0.55 on both."
            ),
            "orientation_suspect": orientation_suspect,
            "orientation_note": "A raw target AUROC<=0.25 with reversed-score AUROC>=0.75 flags possible target ranking inversion; inspect semantics before changing score orientation.",
            "mapping_version": mapping.version,
            "verified_only": not args.allow_unverified_mapping,
            "target_label_oracle_fields_are_diagnostic_only": True,
        }
        run.write_json("sanity_gate.json", gate)
        print("\n", table[[
            "model", "macro_f1", "balanced_accuracy", "auprc", "auroc", "fpr", "fnr",
            "target_fraction_above_source_threshold", "diagnostic_target_oracle_macro_f1",
            "diagnostic_target_auroc_reversed", "diagnostic_orientation_suspect",
        ]].to_string(index=False))
        print("\nTRANSFER SANITY STATUS:", status)
        if args.allow_unverified_mapping:
            print("WARNING: candidate/unverified mapping was used; result is diagnostic only.")
        print("Run artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
