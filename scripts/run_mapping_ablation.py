#!/usr/bin/env python
"""Cheap target-labelled mapping ablation for semantic debugging only.

This script is intentionally NOT a final model-selection tool. It helps identify
candidate mappings whose inclusion flips ranking/orientation or creates severe score
shift. Final feature inclusion must be justified by semantics/unit verification, not
by target-test optimization.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

from ids.data.canonical_mapping import align_frames_with_mapping, resolve_mapping
from ids.data.loaders import load_csv
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.score_shift import build_score_shift_diagnostic, positive_class_score
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


def _make_model(name: str, seed: int):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=seed)
    if name == "SGDClassifier":
        return SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=2000, tol=1e-4, random_state=seed)
    raise ValueError(name)


def _load_groups(path: str, resolved: list[str]) -> list[tuple[str, list[str], str]]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = []
    for name, spec in (cfg.get("groups") or {}).items():
        spec = spec or {}
        include = spec.get("include")
        exclude = set(spec.get("exclude") or [])
        selected = [f for f in resolved if (include is None or f in include) and f not in exclude]
        if selected:
            out.append((name, selected, str(spec.get("description", ""))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--target-csv", required=True)
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--groups", default="configs/features/ablation_groups.yaml")
    ap.add_argument("--allow-unverified-mapping", action="store_true")
    ap.add_argument("--mode", choices=["groups", "loo", "both"], default="groups")
    ap.add_argument("--models", nargs="+", choices=["LogisticRegression", "RandomForest", "SGDClassifier"], default=["LogisticRegression"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-source", type=int, default=100000)
    ap.add_argument("--max-target", type=int, default=50000)
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    set_global_seed(args.seed)

    source_raw = _stratified_cap(load_csv(args.source_csv), args.source_name, args.max_source, args.seed)
    target_raw = _stratified_cap(load_csv(args.target_csv), args.target_name, args.max_target, args.seed + 1)
    mapping = resolve_mapping(
        source_raw.columns, target_raw.columns, args.source_name, args.target_name,
        mapping_path=args.mapping, verified_only=not args.allow_unverified_mapping,
    )
    if not mapping.features:
        raise SystemExit("No usable mapping.")
    source, target = align_frames_with_mapping(source_raw, target_raw, mapping)
    resolved = [f.canonical for f in mapping.features]

    subsets: list[tuple[str, list[str], str]] = []
    if args.mode in {"groups", "both"}:
        subsets.extend(_load_groups(args.groups, resolved))
    if args.mode in {"loo", "both"}:
        subsets.extend([
            (f"loo_without_{f}", [x for x in resolved if x != f], f"Leave-one-out diagnostic removing {f}")
            for f in resolved if len(resolved) > 1
        ])

    out_dir = Path(args.output_dir or f"results/diagnostics/mapping_ablation_{args.source_name}_to_{args.target_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for subset_name, features, description in subsets:
        s = source[features + ["Label"]].copy()
        t = target[features + ["Label"]].copy()
        split = prepare_clean_source_target(
            s, t, args.source_name, args.target_name,
            seed=args.seed, threshold=0.0, feature_alignment="exact",
        )
        for model_name in args.models:
            model = _make_model(model_name, args.seed)
            t0 = time.time()
            model.fit(split.X_source_train, split.y_source_train)
            val_score, meta = positive_class_score(model, split.X_source_val)
            threshold, val_metric = choose_binary_threshold(
                split.y_source_val, val_score, strategy="source_val_macro_f1"
            )
            score, _ = positive_class_score(model, split.X_target_test)
            pred = (score >= threshold).astype(int)
            task = {**binary(split.y_target_test, pred, score), **binary_confusion(split.y_target_test, pred)}
            diag = build_score_shift_diagnostic(
                source_val_score=val_score,
                source_val_labels=split.y_source_val,
                source_threshold=threshold,
                target_score=score,
                target_labels=split.y_target_test,
            )
            rows.append({
                "subset": subset_name,
                "description": description,
                "model": model_name,
                "requested_feature_count": len(features),
                "selected_feature_count": len(split.selected_feature_names),
                "selected_features": "|".join(split.selected_feature_names),
                "source_threshold": threshold,
                "source_threshold_metric": val_metric,
                "score_method": meta.method,
                "train_seconds": round(time.time() - t0, 3),
                **task,
                **diag,
            })

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "mapping_ablation.csv", index=False)
    metadata = {
        "diagnostic_only": True,
        "target_labels_used_for_evaluation_and_oracle_diagnostics": True,
        "must_not_select_final_mapping_by_target_test_score": True,
        "mapping_version": mapping.version,
        "resolved_features": resolved,
        "mode": args.mode,
        "models": args.models,
    }
    (out_dir / "README_DIAGNOSTIC_ONLY.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cols = [
        "subset", "model", "selected_feature_count", "auroc", "macro_f1",
        "balanced_accuracy", "target_fraction_above_source_threshold",
        "diagnostic_target_oracle_macro_f1", "diagnostic_target_auroc_reversed",
        "diagnostic_orientation_suspect",
    ]
    print("\nIDS v0.4.2 MAPPING ABLATION — DIAGNOSTIC ONLY")
    print("=" * 62)
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    print("\nDo NOT choose the final mapping by the best target-test row.")
    print("Use this table to identify mappings that deserve semantic/unit review.")
    print("Saved diagnostics to:", out_dir.resolve())


if __name__ == "__main__":
    main()
