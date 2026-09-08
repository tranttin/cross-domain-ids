#!/usr/bin/env python
"""E4: multi-source (2017+2018) -> CICIoT2023 clean transfer.

The two source domains are first aligned to the same canonical feature names and
concatenated. Target labels are never used for training/model selection. For final
runs, pass --target-test-csv to use a frozen external target holdout.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
from types import SimpleNamespace
import numpy as np
import pandas as pd

from ids.data.loaders import load_csv
from ids.data.canonical_mapping import resolve_mapping, align_frames_with_mapping
from ids.preprocessing.cross_domain import (
    prepare_clean_source_target,
    prepare_clean_source_target_fixed_test,
)
from ids.models.clean_dann import train_clean_dann
from ids.models.tabular import shared_layer_weights
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.thresholds import choose_binary_threshold
from ids.evaluation.unsupervised_thresholds import choose_unsupervised_target_threshold
from ids.experiments.tracker import ExperimentTracker
from ids.utils.seed import set_global_seed

# Reuse the validated v0.4.8/v0.5.0 source-only MLP-BN training routine.
from run_transfer_clean import train_source_only


def _aligned(df_source, source_name, target_df, target_name, mapping_path, verified_only):
    m = resolve_mapping(
        df_source.columns, target_df.columns, source_name, target_name,
        mapping_path=mapping_path, verified_only=verified_only,
    )
    s, t = align_frames_with_mapping(df_source, target_df, m)
    return m, s, t


def _metrics(y, score, threshold):
    score = np.asarray(score, dtype=float)
    pred = (score >= float(threshold)).astype(int)
    return {**binary(y, pred, score), **binary_confusion(y, pred)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-a-csv", required=True)
    ap.add_argument("--source-a-name", default="2017")
    ap.add_argument("--source-b-csv", required=True)
    ap.add_argument("--source-b-name", default="2018")
    ap.add_argument("--target-csv", required=True, help="Target adaptation/dev CSV")
    ap.add_argument("--target-test-csv", help="Frozen target final-test CSV")
    ap.add_argument("--target-name", default="CICIoT2023")
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--allow-unverified-mapping", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-d", type=float, default=0.01)
    ap.add_argument("--domain-loss-weight", type=float, default=0.5)
    ap.add_argument("--source-lr", type=float, default=1e-3)
    ap.add_argument("--dann-lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--target-threshold-strategy", choices=["source", "gmm_logit", "valley_logit", "source_prior_quantile"], default="gmm_logit")
    ap.add_argument("--only", choices=["all", "source_only", "dann"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-root", default="results")
    args = ap.parse_args()
    set_global_seed(args.seed)
    if args.smoke:
        args.epochs = min(args.epochs, 8)
        args.patience = min(args.patience, 3)

    a = load_csv(args.source_a_csv)
    b = load_csv(args.source_b_csv)
    target_dev = load_csv(args.target_csv)
    target_test_raw = load_csv(args.target_test_csv) if args.target_test_csv else None
    verified_only = not args.allow_unverified_mapping

    ma, aa, ta = _aligned(a, args.source_a_name, target_dev, args.target_name, args.mapping, verified_only)
    mb, bb, tb = _aligned(b, args.source_b_name, target_dev, args.target_name, args.mapping, verified_only)
    common = sorted(set(f.canonical for f in ma.features) & set(f.canonical for f in mb.features))
    if len(common) < 5:
        raise SystemExit(f"Only {len(common)} shared canonical features across both sources and target")
    aa = aa[common + ["Label"]]
    bb = bb[common + ["Label"]]
    ta = ta[common + ["Label"]]
    # Target canonical values are the same columns for both mappings; use one copy.
    merged = pd.concat([aa, bb], ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    if target_test_raw is not None:
        _ma2, _sa2, tt = _aligned(a, args.source_a_name, target_test_raw, args.target_name, args.mapping, verified_only)
        tt = tt[common + ["Label"]]
        split = prepare_clean_source_target_fixed_test(
            merged, ta, tt, "2017+2018", args.target_name,
            seed=args.seed, threshold=0.01, feature_alignment="exact",
        )
    else:
        split = prepare_clean_source_target(
            merged, ta, "2017+2018", args.target_name,
            seed=args.seed, threshold=0.01, feature_alignment="exact",
        )

    protocol = "v0.5.0_E4_multisource_smoke" if args.smoke else "v0.5.0_E4_multisource_final"
    if args.allow_unverified_mapping:
        protocol += "_UNVERIFIED_DIAGNOSTIC"
    cfg = vars(args).copy(); cfg.update({"common_canonical_features": common})

    with ExperimentTracker(
        experiment="paper_E4_multisource", protocol=protocol, seed=args.seed,
        config=cfg, output_root=args.output_root,
    ) as run:
        run.save_feature_names(split.selected_feature_names)
        rows = []
        source_model = None
        if args.only in {"all", "source_only"}:
            source_model, _hist, weights, thr_res, m = train_source_only(
                run, split, args.epochs, args.batch_size, args.source_lr, args.patience,
                args.target_threshold_strategy, args.seed, "mlp_bn_fair",
            )
            row = {
                "paper_stage": "E4", "paper_table": "T5_MultiSource",
                "model": "MLP-BN-Canonical", "train_domain": "2017+2018",
                "test_domain": args.target_name, "task": "binary", "adaptation": "none",
                "mapping_version": ma.version, "mapping_verified_only": verified_only,
                "feature_count": len(split.selected_feature_names), "encoder": "mlp_bn_fair",
                "target_labels_in_training": False, "lambda_d": np.nan,
                **m,
            }
            rows.append(row); run.record_metrics(row)

        if args.only in {"all", "dann"}:
            if source_model is None:
                source_model, _hist, _weights, _thr_res, _m = train_source_only(
                    run, split, args.epochs, args.batch_size, args.source_lr, args.patience,
                    args.target_threshold_strategy, args.seed, "mlp_bn_fair",
                )
            t0 = time.time()
            result = train_clean_dann(
                split.X_source_train, split.y_source_train,
                split.X_source_val, split.y_source_val,
                split.X_target_unlabeled,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.dann_lr,
                lambda_d=args.lambda_d, domain_loss_weight=args.domain_loss_weight,
                class_weight_mode="balanced", patience=args.patience, seed=args.seed,
                verbose=1, abort_on_collapse=False, encoder="mlp_bn_fair",
                initial_shared_weights=shared_layer_weights(source_model),
                freeze_batchnorm=True,
            )
            val_cls, _ = result.model.predict(split.X_source_val, verbose=0)
            src_thr, src_val_metric = choose_binary_threshold(
                split.y_source_val, val_cls[:, 1], strategy="source_val_macro_f1"
            )
            adapt_cls, _ = result.model.predict(split.X_target_unlabeled, verbose=0)
            test_cls, _ = result.model.predict(split.X_target_test, verbose=0)
            if args.target_threshold_strategy == "source":
                threshold = float(src_thr); status = "source"; adapt_rate = float(np.mean(adapt_cls[:,1] >= threshold))
            else:
                tr = choose_unsupervised_target_threshold(
                    adapt_cls[:, 1], source_scores=val_cls[:, 1], source_labels=split.y_source_val,
                    source_threshold=src_thr, strategy=args.target_threshold_strategy,
                )
                threshold = float(tr.threshold); status = tr.status; adapt_rate = float(tr.target_adapt_positive_rate)
            m = _metrics(split.y_target_test, test_cls[:,1], threshold)
            row = {
                "paper_stage": "E4", "paper_table": "T5_MultiSource",
                "model": "MLP-BN-DANN", "train_domain": "2017+2018",
                "test_domain": args.target_name, "task": "binary", "adaptation": "UDA",
                "mapping_version": ma.version, "mapping_verified_only": verified_only,
                "feature_count": len(split.selected_feature_names), "encoder": "mlp_bn_fair",
                "target_labels_in_training": False, "lambda_d": args.lambda_d,
                "domain_loss_weight": args.domain_loss_weight,
                "best_epoch": result.best_epoch,
                "best_source_val_macro_f1": result.best_val_macro_f1,
                "source_val_decision_threshold": float(src_thr),
                "decision_threshold": threshold,
                "threshold_source_val_metric": float(src_val_metric),
                "target_threshold_strategy": args.target_threshold_strategy,
                "target_adapt_positive_rate_at_threshold": adapt_rate,
                "threshold_status": status,
                "final_val_domain_accuracy": result.history.get("val_domain_accuracy", [np.nan])[-1],
                "train_seconds": round(time.time() - t0, 3),
                **m,
            }
            rows.append(row); run.record_metrics(row)

        out = pd.DataFrame(rows)
        run.save_dataframe("table.csv", out)
        print("\n", out.to_string(index=False))
        print("Run artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
