#!/usr/bin/env python
"""v0.4.5 generic clean transfer runner with semantic mapping + label-free target calibration.

Use this only AFTER `diagnose_domain_shift.py` and `run_transfer_baselines.py`.
Final paper runs require verified mappings. `--allow-unverified-mapping` and `--smoke`
are engineering diagnostics and are tagged as non-final in the run log.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical

from ids.data.loaders import load_csv
from ids.data.canonical_mapping import resolve_mapping, align_frames_with_mapping, mapping_dataframe
from ids.preprocessing.cross_domain import prepare_clean_source_target, prepare_clean_source_target_fixed_test, binary_labels_for_dataset
from ids.models.hybrid import build_legacy_binary_hybrid
from ids.models.tabular import build_mlp_binary, build_mlp_fair_binary, build_mlp_bn_fair_binary, shared_layer_weights
from ids.models.clean_dann import train_clean_dann
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.thresholds import choose_binary_threshold
from ids.evaluation.unsupervised_thresholds import choose_unsupervised_target_threshold
from ids.experiments.tracker import ExperimentTracker
from ids.training.imbalance import balanced_class_weights, class_weight_vector
from ids.utils.seed import set_global_seed


def _metrics(y, prob, threshold: float):
    score = np.asarray(prob)[:, 1]
    pred = (score >= float(threshold)).astype(int)
    return {**binary(y, pred, score), **binary_confusion(y, pred)}


def _cap_source(df: pd.DataFrame, name: str, n: int | None, seed: int) -> pd.DataFrame:
    if not n or n >= len(df):
        return df
    y = binary_labels_for_dataset(df, name)
    rng = np.random.default_rng(seed)
    out = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        k = max(1, int(round(n * len(idx) / len(df))))
        pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
        out.append(df.iloc[pick])
    return pd.concat(out).sample(frac=1, random_state=seed).reset_index(drop=True)


def _cap_target_unlabelled(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    # Deliberately random, NOT stratified by target class labels.
    if not n or n >= len(df):
        return df
    return df.sample(n=int(n), random_state=seed).reset_index(drop=True)


class SourceValThresholdMonitor(tf.keras.callbacks.Callback):
    """Threshold-aware source validation monitor.

    v0.4.4 fix: class weighting and cross-domain score shift make 0.5 an invalid
    checkpoint threshold. Each epoch chooses the threshold ONLY on labelled source
    validation. Target scores are diagnostics and never select/checkpoint the model.
    """
    def __init__(self, X_val, y_val, X_target_monitor, input_already_shaped: bool = False):
        super().__init__()
        if input_already_shaped:
            self.Xv = np.asarray(X_val, dtype=np.float32)
            self.Xt = np.asarray(X_target_monitor[:4096], dtype=np.float32)
        else:
            self.Xv = np.asarray(X_val, dtype=np.float32)[..., None]
            self.Xt = np.asarray(X_target_monitor[:4096], dtype=np.float32)[..., None]
        self.yv = np.asarray(y_val).astype(int)

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import balanced_accuracy_score
        logs = logs if logs is not None else {}
        pv = self.model.predict(self.Xv, verbose=0)
        threshold, macro = choose_binary_threshold(
            self.yv, pv[:, 1], strategy="source_val_macro_f1"
        )
        pred = (pv[:, 1] >= threshold).astype(int)
        bal = float(balanced_accuracy_score(self.yv, pred))
        pt = self.model.predict(self.Xt, verbose=0)
        target_rate = float(np.mean(pt[:, 1] >= threshold))
        logs["val_macro_f1_opt"] = float(macro)
        logs["val_balanced_accuracy_opt"] = bal
        logs["val_decision_threshold"] = float(threshold)
        logs["target_predicted_positive_rate_at_source_threshold"] = target_rate


def train_source_only(run, split, epochs, batch_size, lr, patience, target_threshold_strategy, seed, encoder):
    if encoder == "cnn_bilstm_attention":
        model = build_legacy_binary_hybrid(split.X_source_train.shape[1], 1, lr=lr)
        x_train = split.X_source_train[..., None]
        x_val = split.X_source_val[..., None]
        x_adapt = split.X_target_unlabeled[..., None]
        x_test = split.X_target_test[..., None]
    elif encoder == "mlp":
        model = build_mlp_binary(split.X_source_train.shape[1], lr=lr)
        x_train = split.X_source_train
        x_val = split.X_source_val
        x_adapt = split.X_target_unlabeled
        x_test = split.X_target_test
    elif encoder == "mlp_fair":
        model = build_mlp_fair_binary(split.X_source_train.shape[1], lr=lr)
        x_train = split.X_source_train
        x_val = split.X_source_val
        x_adapt = split.X_target_unlabeled
        x_test = split.X_target_test
    elif encoder == "mlp_bn_fair":
        model = build_mlp_bn_fair_binary(split.X_source_train.shape[1], lr=lr)
        x_train = split.X_source_train
        x_val = split.X_source_val
        x_adapt = split.X_target_unlabeled
        x_test = split.X_target_test
    else:
        raise ValueError("encoder must be 'cnn_bilstm_attention', 'mlp', 'mlp_fair', or 'mlp_bn_fair'")
    weights = balanced_class_weights(split.y_source_train)
    sw = class_weight_vector(split.y_source_train, weights)
    # The callback supports both sequence and tabular inputs by receiving already
    # shaped arrays; for tabular MLP no synthetic channel dimension is introduced.
    monitor = SourceValThresholdMonitor(
        x_val, split.y_source_val, x_adapt, input_already_shaped=True
    )
    callbacks = [
        monitor,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1_opt", mode="max", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1_opt", mode="max", factor=0.3, patience=2, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(str(run.run_dir / "history_source_only.csv")),
    ]
    t0 = time.time()
    hist = model.fit(
        x_train, to_categorical(split.y_source_train, 2),
        sample_weight=sw,
        validation_data=(x_val, to_categorical(split.y_source_val, 2)),
        epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=callbacks, verbose=1,
    )
    val_prob = model.predict(x_val, verbose=0)
    threshold, val_metric = choose_binary_threshold(
        split.y_source_val, val_prob[:, 1], strategy="source_val_macro_f1"
    )
    adapt_prob = model.predict(x_adapt, verbose=0)
    threshold_result = choose_unsupervised_target_threshold(
        target_threshold_strategy,
        target_adapt_score=adapt_prob[:, 1],
        source_threshold=threshold,
        source_positive_prevalence=float(np.mean(split.y_source_train)),
        seed=seed,
    )
    prob = model.predict(x_test, verbose=0)
    m = _metrics(split.y_target_test, prob, threshold_result.threshold)
    mh = hist.history.get("val_macro_f1_opt", [])
    return model, hist, weights, threshold_result, {
        **m,
        "train_seconds": round(time.time() - t0, 3),
        "best_epoch": int(np.argmax(mh) + 1) if mh else None,
        "best_source_val_macro_f1": float(np.max(mh)) if mh else None,
        "source_val_decision_threshold": float(threshold),
        "decision_threshold": float(threshold_result.threshold),
        "threshold_source_val_metric": float(val_metric),
        "target_threshold_strategy": target_threshold_strategy,
        "threshold_fit_pool": "source_validation" if target_threshold_strategy == "source" else "target_unlabelled_adaptation",
        "threshold_target_labels_used": False,
        "target_adapt_positive_rate_at_threshold": float(threshold_result.target_adapt_positive_rate),
        "threshold_status": threshold_result.status,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--target-csv", required=True, help="Target unlabeled adaptation/dev CSV.")
    ap.add_argument("--target-test-csv", help="Optional externally frozen target test CSV for final-paper evaluation.")
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--allow-unverified-mapping", action="store_true")
    ap.add_argument("--min-features", type=int, default=5)
    ap.add_argument("--only", choices=["all", "source_only", "dann"], default="all")
    ap.add_argument("--encoder", choices=["cnn_bilstm_attention", "mlp", "mlp_fair", "mlp_bn_fair"], default="cnn_bilstm_attention", help="mlp_bn_fair is the v0.4.8 staged source-pretrained/frozen-BN DANN fairness control; mlp preserves v0.4.5 diagnostics.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-epochs", type=int, help="Hard cap useful for controlled diagnostics.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lambda-d", type=float, default=0.1)
    ap.add_argument("--domain-loss-weight", type=float, default=0.5)
    ap.add_argument("--source-lr", type=float, default=1e-3)
    ap.add_argument("--dann-lr", type=float, default=None, help="If omitted: match --source-lr for MLP controls; use 1e-4 for CNN-BiLSTM-Attention.")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--sample-source", type=int)
    ap.add_argument("--sample-target", type=int)
    ap.add_argument("--smoke", action="store_true", help="Engineering-only: up to 8 epochs, <=20k source/target unless overridden.")
    ap.add_argument("--target-threshold-strategy", choices=["source", "gmm_logit", "valley_logit", "source_prior_quantile"], default="source", help="Threshold is fit on source validation or target UNLABELLED adaptation scores only; never target-test labels.")
    ap.add_argument("--output-root", default="results")
    args = ap.parse_args()
    set_global_seed(args.seed)

    epochs = min(args.epochs, args.max_epochs) if args.max_epochs else args.epochs
    if args.smoke:
        epochs = min(epochs, 8)
        args.sample_source = args.sample_source or 20_000
        args.sample_target = args.sample_target or 20_000
        args.patience = min(args.patience, 3)

    effective_dann_lr = args.dann_lr
    if effective_dann_lr is None:
        effective_dann_lr = args.source_lr if args.encoder in {"mlp", "mlp_fair", "mlp_bn_fair"} else 1e-4

    source = _cap_source(load_csv(args.source_csv), args.source_name, args.sample_source, args.seed)
    target = _cap_target_unlabelled(load_csv(args.target_csv), args.sample_target, args.seed + 1)
    target_test_raw = load_csv(args.target_test_csv) if args.target_test_csv else None
    mapping = resolve_mapping(
        source.columns, target.columns, args.source_name, args.target_name,
        mapping_path=args.mapping, verified_only=not args.allow_unverified_mapping,
    )
    if len(mapping.features) < args.min_features:
        mode = "verified" if not args.allow_unverified_mapping else "candidate"
        raise SystemExit(
            f"Only {len(mapping.features)} {mode} mapped features. Run diagnose_domain_shift.py and verify the mapping."
        )
    source, target = align_frames_with_mapping(source, target, mapping)
    if target_test_raw is not None:
        _source_for_test, target_test = align_frames_with_mapping(
            _cap_source(load_csv(args.source_csv), args.source_name, args.sample_source, args.seed),
            target_test_raw, mapping,
        )
        split = prepare_clean_source_target_fixed_test(
            source, target, target_test, args.source_name, args.target_name,
            seed=args.seed, threshold=0.01, feature_alignment="exact",
        )
    else:
        split = prepare_clean_source_target(
            source, target, args.source_name, args.target_name,
            seed=args.seed, threshold=0.01, feature_alignment="exact",
        )

    protocol = "v0.5.0_clean_transfer_smoke" if args.smoke else "v0.5.0_clean_transfer_final"
    if args.allow_unverified_mapping:
        protocol += "_UNVERIFIED_DIAGNOSTIC"
    cfg = vars(args).copy(); cfg.update({"effective_epochs": epochs, "effective_dann_lr": effective_dann_lr, "mapping_version": mapping.version})

    with ExperimentTracker(
        experiment="generic_clean_transfer", protocol=protocol,
        seed=args.seed, config=cfg, output_root=args.output_root,
    ) as run:
        mapping_dataframe(mapping).to_csv(run.run_dir / "feature_mapping.csv", index=False)
        run.save_feature_names(split.selected_feature_names, "selected_features.json")
        run.logger.info(
            "direction=%s->%s mapped=%d selected=%d verified_only=%s smoke=%s",
            args.source_name, args.target_name, len(mapping.features), len(split.selected_feature_names),
            not args.allow_unverified_mapping, args.smoke,
        )
        run.logger.info(
            "source prevalence train=%.4f val=%.4f | target-test prevalence=%.4f (evaluation only)",
            float(np.mean(split.y_source_train)), float(np.mean(split.y_source_val)), float(np.mean(split.y_target_test)),
        )
        rows = []
        source_model_for_staged_dann = None

        if args.only in {"all", "source_only"}:
            model, hist, weights, threshold_result, m = train_source_only(
                run, split, epochs, args.batch_size, args.source_lr, args.patience,
                args.target_threshold_strategy, args.seed, args.encoder
            )
            run.write_json("threshold_source_only.json", threshold_result.to_dict())
            if args.encoder == "mlp_bn_fair":
                source_model_for_staged_dann = model
            row = {
                "model": (("MLP-BN-Fair-Canonical-Control" if args.encoder == "mlp_bn_fair" else "MLP-Fair-Canonical-Control") if args.encoder in {"mlp_fair", "mlp_bn_fair"} else "MLP-Canonical-Control") if args.encoder in {"mlp", "mlp_fair", "mlp_bn_fair"} else "CNN-BiLSTM-Attention", "train_domain": args.source_name,
                "test_domain": args.target_name, "task": "binary", "adaptation": "none",
                "target_labels_in_training": False, "mapping_version": mapping.version,
                "mapping_verified_only": not args.allow_unverified_mapping,
                "feature_count": len(split.selected_feature_names), "class_weight_mode": "balanced",
                "source_weight_0": weights.get(0), "source_weight_1": weights.get(1),
                "source_threshold_strategy": "source_val_macro_f1", "encoder": args.encoder, "smoke": args.smoke, **m,
            }
            rows.append(row); run.record_metrics(row)

        if args.only in {"all", "dann"}:
            # v0.4.8 staged fairness: the successful BN-based tabular encoder is
            # source-pretrained first. DANN starts from the exact same task weights,
            # and BatchNorm moving statistics are frozen during target adaptation.
            if args.encoder == "mlp_bn_fair" and source_model_for_staged_dann is None:
                run.logger.info("mlp_bn_fair requires source pretraining; running an internal source-only pretrain before DANN.")
                source_model_for_staged_dann, _hist0, _w0, _thr0, _m0 = train_source_only(
                    run, split, epochs, args.batch_size, args.source_lr, args.patience,
                    args.target_threshold_strategy, args.seed, args.encoder
                )
            staged_weights = shared_layer_weights(source_model_for_staged_dann) if source_model_for_staged_dann is not None else None
            t0 = time.time()
            result = train_clean_dann(
                split.X_source_train, split.y_source_train,
                split.X_source_val, split.y_source_val,
                split.X_target_unlabeled,
                epochs=epochs, batch_size=args.batch_size,
                lr=effective_dann_lr, lambda_d=args.lambda_d,
                domain_loss_weight=args.domain_loss_weight,
                class_weight_mode="balanced", patience=args.patience,
                seed=args.seed, verbose=1, abort_on_collapse=False, encoder=args.encoder,
                initial_shared_weights=staged_weights,
                freeze_batchnorm=(args.encoder == "mlp_bn_fair"),
            )
            run.save_history(result.history, "history_clean_dann.csv")
            if args.encoder in {"mlp", "mlp_fair", "mlp_bn_fair"}:
                x_val_dann = split.X_source_val
                x_adapt_dann = split.X_target_unlabeled
                x_test_dann = split.X_target_test
            else:
                x_val_dann = split.X_source_val[..., None]
                x_adapt_dann = split.X_target_unlabeled[..., None]
                x_test_dann = split.X_target_test[..., None]
            val_cls, _ = result.model.predict(x_val_dann, verbose=0)
            threshold, val_metric = choose_binary_threshold(
                split.y_source_val, val_cls[:, 1], strategy="source_val_macro_f1"
            )
            adapt_prob, _ = result.model.predict(x_adapt_dann, verbose=0)
            threshold_result = choose_unsupervised_target_threshold(
                args.target_threshold_strategy,
                target_adapt_score=adapt_prob[:, 1],
                source_threshold=threshold,
                source_positive_prevalence=float(np.mean(split.y_source_train)),
                seed=args.seed,
            )
            run.write_json("threshold_clean_dann.json", threshold_result.to_dict())
            prob, _ = result.model.predict(x_test_dann, verbose=0)
            m = _metrics(split.y_target_test, prob, threshold_result.threshold)
            weights = result.source_class_weights or {}
            row = {
                "model": (("MLP-BN-Fair-DANN-Canonical-Control" if args.encoder == "mlp_bn_fair" else "MLP-Fair-DANN-Canonical-Control") if args.encoder in {"mlp_fair", "mlp_bn_fair"} else "MLP-DANN-Canonical-Control") if args.encoder in {"mlp", "mlp_fair", "mlp_bn_fair"} else "Clean-DANN", "train_domain": args.source_name,
                "test_domain": args.target_name, "task": "binary", "adaptation": "UDA",
                "target_labels_in_training": False, "mapping_version": mapping.version,
                "mapping_verified_only": not args.allow_unverified_mapping,
                "feature_count": len(split.selected_feature_names), "class_weight_mode": "balanced",
                "source_weight_0": weights.get(0), "source_weight_1": weights.get(1),
                "source_threshold_strategy": "source_val_macro_f1",
                "source_val_decision_threshold": threshold,
                "decision_threshold": threshold_result.threshold,
                "threshold_source_val_metric": val_metric,
                "target_threshold_strategy": args.target_threshold_strategy,
                "threshold_fit_pool": "source_validation" if args.target_threshold_strategy == "source" else "target_unlabelled_adaptation",
                "threshold_target_labels_used": False,
                "target_adapt_positive_rate_at_threshold": threshold_result.target_adapt_positive_rate,
                "threshold_status": threshold_result.status,
                "lambda_d": args.lambda_d,
                "domain_loss_weight": args.domain_loss_weight, "encoder": args.encoder, "dann_lr": effective_dann_lr,
                "dann_init": "source_pretrained" if args.encoder == "mlp_bn_fair" else "random",
                "batchnorm_frozen_during_dann": bool(args.encoder == "mlp_bn_fair"),
                "best_epoch": result.best_epoch, "best_source_val_macro_f1": result.best_val_macro_f1,
                "final_val_domain_accuracy": result.history.get("val_domain_accuracy", [np.nan])[-1],
                "final_unlabelled_target_predicted_positive_rate": result.history.get("target_predicted_positive_rate_at_source_threshold", result.history.get("target_predicted_positive_rate", [np.nan]))[-1],
                "train_seconds": round(time.time() - t0, 3), "smoke": args.smoke, **m,
            }
            rows.append(row); run.record_metrics(row)

        table = pd.DataFrame(rows)
        run.save_dataframe("table.csv", table)
        print("\n", table.to_string(index=False) if len(table) else "<no rows>")
        if args.smoke or args.allow_unverified_mapping:
            print("\nWARNING: this is an engineering diagnostic, not final paper evidence.")
        print("Run artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
