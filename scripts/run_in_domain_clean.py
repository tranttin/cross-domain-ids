#!/usr/bin/env python
"""Clean binary in-domain benchmark for the journal paper (E1).

Runs five classical baselines plus CNN--BiLSTM--Attention on one prepared dataset.
All models use the same train/validation/test split and training-only preprocessing.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from ids.data.loaders import load_csv
from ids.preprocessing.in_domain import preprocess_clean_binary_in_domain, preprocess_clean_binary_in_domain_fixed_test
from ids.models.classical import build_clean_binary_models
from ids.models.hybrid import build_legacy_binary_hybrid
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.thresholds import choose_binary_threshold
from ids.evaluation.score_shift import positive_class_score
from ids.experiments.tracker import ExperimentTracker
from ids.training.imbalance import balanced_class_weights, class_weight_vector
from ids.utils.seed import set_global_seed


def evaluate(y, score, threshold):
    score = np.asarray(score, dtype=float)
    pred = (score >= float(threshold)).astype(int)
    return {**binary(y, pred, score), **binary_confusion(y, pred)}


def stratified_cap(df: pd.DataFrame, dataset_name: str, n: int | None, seed: int):
    if not n or len(df) <= n:
        return df
    from ids.preprocessing.cross_domain import binary_labels_for_dataset
    y = binary_labels_for_dataset(df, dataset_name)
    rng = np.random.default_rng(seed)
    parts = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        k = max(1, int(round(n * len(idx) / len(df))))
        pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
        parts.append(df.iloc[pick])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


class ThresholdAwareMonitor(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = np.asarray(x_val, dtype=np.float32)[..., None]
        self.y_val = np.asarray(y_val, dtype=np.int32)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        prob = self.model.predict(self.x_val, verbose=0)[:, 1]
        thr, mf1 = choose_binary_threshold(self.y_val, prob, strategy="source_val_macro_f1")
        pred = (prob >= thr).astype(int)
        from sklearn.metrics import balanced_accuracy_score
        logs["val_macro_f1_opt"] = float(mf1)
        logs["val_balanced_accuracy_opt"] = float(balanced_accuracy_score(self.y_val, pred))
        logs["val_decision_threshold"] = float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--test-csv", help="Optional externally frozen test CSV.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=["all", "classical", "cnn"], default="all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--variance-threshold", type=float, default=0.01)
    ap.add_argument("--sample", type=int)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-root", default="results")
    args = ap.parse_args()
    set_global_seed(args.seed)

    if args.smoke:
        args.sample = args.sample or 20_000
        args.epochs = min(args.epochs, 5)
        args.patience = min(args.patience, 2)

    data = stratified_cap(load_csv(args.csv), args.dataset_name, args.sample, args.seed)
    if args.test_csv:
        test_data = load_csv(args.test_csv)
        split = preprocess_clean_binary_in_domain_fixed_test(
            data, test_data, args.dataset_name, seed=args.seed,
            variance_threshold=args.variance_threshold,
        )
    else:
        split = preprocess_clean_binary_in_domain(
            data, args.dataset_name, seed=args.seed,
            variance_threshold=args.variance_threshold,
        )
    protocol = "v0.5.0_clean_binary_in_domain_smoke" if args.smoke else "v0.5.0_clean_binary_in_domain"
    cfg = vars(args).copy()

    with ExperimentTracker(
        experiment="paper_E1_in_domain", protocol=protocol, seed=args.seed,
        config=cfg, output_root=args.output_root,
    ) as run:
        run.save_feature_names(split.selected_feature_names)
        run.logger.info(
            "dataset=%s train=%s val=%s test=%s prevalence(train/val/test)=%.4f/%.4f/%.4f",
            args.dataset_name, split.X_train.shape, split.X_val.shape, split.X_test.shape,
            np.mean(split.y_train), np.mean(split.y_val), np.mean(split.y_test),
        )
        rows = []

        if args.only in {"all", "classical"}:
            for name, model in build_clean_binary_models(args.seed).items():
                t0 = time.time()
                model.fit(split.X_train, split.y_train)
                val_score, _ = positive_class_score(model, split.X_val)
                thr, val_metric = choose_binary_threshold(
                    split.y_val, val_score, strategy="source_val_macro_f1"
                )
                test_score, _ = positive_class_score(model, split.X_test)
                row = {
                    "paper_stage": "E1", "paper_table": "T2_InDomain",
                    "model": name, "train_domain": args.dataset_name,
                    "test_domain": args.dataset_name, "task": "binary",
                    "adaptation": "none", "threshold_strategy": "source_val_macro_f1",
                    "decision_threshold": float(thr),
                    "threshold_source_val_metric": float(val_metric),
                    "train_seconds": round(time.time() - t0, 3),
                    **evaluate(split.y_test, test_score, thr),
                }
                rows.append(row); run.record_metrics(row)

        if args.only in {"all", "cnn"}:
            model = build_legacy_binary_hybrid(split.X_train.shape[1], 1, lr=args.lr)
            weights = balanced_class_weights(split.y_train)
            sw = class_weight_vector(split.y_train, weights)
            monitor = ThresholdAwareMonitor(split.X_val, split.y_val)
            callbacks = [
                monitor,
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_macro_f1_opt", mode="max", patience=args.patience,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_macro_f1_opt", mode="max", factor=0.3,
                    patience=2, min_lr=1e-6,
                ),
                tf.keras.callbacks.CSVLogger(str(run.run_dir / "history_cnn.csv")),
            ]
            t0 = time.time()
            hist = model.fit(
                split.X_train[..., None], to_categorical(split.y_train, 2),
                sample_weight=sw,
                validation_data=(split.X_val[..., None], to_categorical(split.y_val, 2)),
                epochs=args.epochs, batch_size=args.batch_size,
                callbacks=callbacks, shuffle=True, verbose=1,
            )
            val_score = model.predict(split.X_val[..., None], verbose=0)[:, 1]
            thr, val_metric = choose_binary_threshold(
                split.y_val, val_score, strategy="source_val_macro_f1"
            )
            test_score = model.predict(split.X_test[..., None], verbose=0)[:, 1]
            h = hist.history.get("val_macro_f1_opt", [])
            row = {
                "paper_stage": "E1", "paper_table": "T2_InDomain",
                "model": "CNN-BiLSTM-Attention", "train_domain": args.dataset_name,
                "test_domain": args.dataset_name, "task": "binary",
                "adaptation": "none", "threshold_strategy": "source_val_macro_f1",
                "decision_threshold": float(thr),
                "threshold_source_val_metric": float(val_metric),
                "best_epoch": int(np.argmax(h) + 1) if h else None,
                "best_source_val_macro_f1": float(np.max(h)) if h else None,
                "source_weight_0": weights.get(0), "source_weight_1": weights.get(1),
                "train_seconds": round(time.time() - t0, 3),
                **evaluate(split.y_test, test_score, thr),
            }
            rows.append(row); run.record_metrics(row)

        table = pd.DataFrame(rows)
        run.save_dataframe("table.csv", table)
        print("\n", table.to_string(index=False))
        print("Run artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
