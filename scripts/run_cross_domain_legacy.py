#!/usr/bin/env python
"""Faithful reproduction of the historical cross-domain notebook.

This runner intentionally preserves methodology problems found during the audit:
1) common feature order from a Python set (unless explicitly overridden),
2) VarianceThreshold + RobustScaler always fit on 2018 train,
3) 2018->2017 is evaluated on full 2017, while 2017->2018 uses 2018 holdout,
4) legacy DANN uses target *class labels* in the classification loss.

Use scripts/run_cross_domain_clean.py for new research.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from ids.data.loaders import load_csv
from ids.preprocessing.cross_domain import prepare_legacy_notebook_protocol
from ids.models.hybrid import build_legacy_binary_hybrid
from ids.models.dann import build_legacy_dann
from ids.models.classical import build_legacy_cross_domain_models
from ids.evaluation.metrics import binary, binary_confusion
from ids.experiments.tracker import ExperimentTracker
from ids.utils.config import load_yaml
from ids.utils.seed import set_global_seed


NOTEBOOK_FEATURE_PREFIX = [
    "Fwd_IAT_Std", "Flow_IAT_Std", "Bwd_PSH_Flags", "Flow_IAT_Max",
    "Bwd_IAT_Min", "Bwd_IAT_Mean", "CWE_Flag_Count", "Idle_Std",
    "Flow_Duration", "Active_Min",
]


def _eval_prob(y_true, prob):
    pred = prob.argmax(axis=1)
    score = prob[:, 1]
    return {**binary(y_true, pred, score), **binary_confusion(y_true, pred)}, pred


def train_eval_hybrid(run, tag, trainX, trainy, testX, testy, cfg):
    deep = cfg["deep"]
    model = build_legacy_binary_hybrid(trainX.shape[1], 1, lr=float(deep.get("learning_rate", 1e-3)))
    if not (run.run_dir / "model_summary_hybrid.txt").exists():
        run.save_model_summary(model, "model_summary_hybrid.txt")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(str(run.run_dir / f"history_{tag}.csv")),
    ]
    t0 = time.time()
    history = model.fit(
        trainX[..., None], to_categorical(trainy, 2),
        epochs=int(deep.get("epochs", 20)), batch_size=int(deep.get("batch_size", 256)),
        validation_split=0.2, shuffle=True, callbacks=callbacks, verbose=1,
    )
    prob = model.predict(testX[..., None], verbose=0)
    metrics, pred = _eval_prob(testy, prob)
    metrics["train_seconds"] = round(time.time() - t0, 3)
    metrics["best_epoch"] = int(np.argmin(history.history["val_loss"]) + 1)
    return metrics, pred


def train_eval_dann(run, tag, sourceX, sourcey, targetX, targety, testX, testy, cfg):
    """Exact supervised-target DANN behavior from the notebook."""
    deep = cfg["deep"]
    X = np.concatenate([sourceX[..., None], targetX[..., None]], axis=0)
    y_cls = to_categorical(np.concatenate([sourcey, targety]), 2)
    y_dom = to_categorical(np.concatenate([
        np.zeros(len(sourceX), dtype=int), np.ones(len(targetX), dtype=int)
    ]), 2)
    model = build_legacy_dann(
        X.shape[1], 1,
        lambda_d=float(deep.get("lambda_d", 1.0)),
        lr=float(deep.get("dann_learning_rate", 1e-4)),
        domain_loss_weight=float(deep.get("domain_loss_weight", 0.5)),
    )
    if not (run.run_dir / "model_summary_dann.txt").exists():
        run.save_model_summary(model, "model_summary_dann.txt")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_cls_output_loss", mode="min", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_cls_output_loss", mode="min", factor=0.3, patience=2, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(str(run.run_dir / f"history_{tag}.csv")),
    ]
    t0 = time.time()
    history = model.fit(
        X, {"cls_output": y_cls, "dom_output": y_dom},
        epochs=int(deep.get("epochs", 20)), batch_size=int(deep.get("batch_size", 256)),
        validation_split=0.2, shuffle=True, callbacks=callbacks, verbose=1,
    )
    prob, _ = model.predict(testX[..., None], verbose=0)
    metrics, pred = _eval_prob(testy, prob)
    metrics["train_seconds"] = round(time.time() - t0, 3)
    val_key = "val_cls_output_loss"
    if val_key in history.history:
        metrics["best_epoch"] = int(np.argmin(history.history[val_key]) + 1)
    return metrics, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/legacy_cross_domain.yaml")
    ap.add_argument("--output-root", default="results")
    ap.add_argument("--seed", type=int, help="Override config seed.")
    ap.add_argument("--feature-order", choices=["legacy_set", "sorted"], help="Override config. 'legacy_set' is notebook-faithful but unstable.")
    ap.add_argument("--feature-order-file", help="Use an exact saved feature order, one feature per line.")
    ap.add_argument("--only", choices=["all", "ml", "cnn", "dann"], default="all")
    ap.add_argument("--cross-only", action="store_true", help="Skip 2018->2018 and 2017->2017 CNN rows.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_global_seed(seed)
    order = cfg["preprocessing"].get("feature_order", "legacy_set")
    if args.feature_order:
        order = args.feature_order
    if args.feature_order_file:
        order = f"file:{args.feature_order_file}"

    protocol = "legacy_notebook_supervised_target_dann"
    with ExperimentTracker(
        experiment="cross_domain_legacy", protocol=protocol, seed=seed,
        config={**cfg, "runtime_feature_order": order}, output_root=args.output_root,
    ) as run:
        if order == "legacy_set" and "PYTHONHASHSEED" not in os.environ:
            run.logger.warning(
                "PYTHONHASHSEED is unset. The notebook-faithful set-based feature order can change between processes. "
                "This run saves the exact order; reuse it later with --feature-order-file."
            )
        df18 = load_csv(cfg["data"]["cic2018_csv"])
        df17 = load_csv(cfg["data"]["cic2017_csv"])
        prep = prepare_legacy_notebook_protocol(
            df17, df18, feature_order=order,
            test_size=float(cfg["data"].get("test_size", 0.2)), seed=seed,
            threshold=float(cfg["preprocessing"].get("variance_threshold", 0.01)),
            quantile_range=tuple(cfg["preprocessing"].get("robust_quantile_range", [5, 95])),
        )
        run.save_feature_names(prep.feature_names, "feature_order_common.json")
        (run.run_dir / "feature_order_common.txt").write_text("\n".join(prep.feature_names) + "\n", encoding="utf-8")
        run.save_feature_names(prep.selected_feature_names, "feature_order_after_variance.json")
        run.logger.info("common_features=%d selected_after_variance=%d", len(prep.feature_names), len(prep.selected_feature_names))
        if prep.feature_names[:10] != NOTEBOOK_FEATURE_PREFIX:
            run.logger.warning(
                "Current common-feature prefix does not match the first 10 columns printed by the historical notebook. "
                "Cross-domain CNN/DANN numbers may differ even though the rest of the protocol is faithful. "
                "This is an unrecoverable legacy set-order issue unless the exact old order is found."
            )
        else:
            run.logger.info("First 10 common features match the historical notebook output.")
        run.logger.info("2018 train/test=%s/%s; 2017 train/test/all=%s/%s/%s", prep.X18_train.shape, prep.X18_test.shape, prep.X17_train.shape, prep.X17_test.shape, prep.X17_all.shape)

        rows = []
        predictions = []

        def record(model, train_domain, test_domain, metrics, pred=None, y_true=None, notes=""):
            row = {
                "model": model, "train_domain": train_domain, "test_domain": test_domain,
                "task": "binary", "notes": notes, **metrics,
            }
            rows.append(row); run.record_metrics(row)
            if pred is not None and y_true is not None:
                predictions.append(pd.DataFrame({
                    "model": model, "train_domain": train_domain, "test_domain": test_domain,
                    "y_true": y_true, "y_pred": pred,
                }))

        if args.only in {"all", "ml"}:
            models = build_legacy_cross_domain_models(seed=seed)
            for name, model in models.items():
                t0 = time.time(); model.fit(prep.X18_train, prep.y18_train)
                pred = model.predict(prep.X17_all)
                m = {**binary(prep.y17_all, pred), **binary_confusion(prep.y17_all, pred), "train_seconds": round(time.time()-t0, 3)}
                record(name, "2018", "2017", m)
            for name, model in build_legacy_cross_domain_models(seed=seed).items():
                t0 = time.time(); model.fit(prep.X17_train, prep.y17_train)
                pred = model.predict(prep.X18_test)
                m = {**binary(prep.y18_test, pred), **binary_confusion(prep.y18_test, pred), "train_seconds": round(time.time()-t0, 3)}
                record(name, "2017", "2018", m)

        if args.only in {"all", "cnn"}:
            cases = []
            if not args.cross_only:
                cases += [
                    ("2018", "2018", prep.X18_train, prep.y18_train, prep.X18_test, prep.y18_test),
                    ("2017", "2017", prep.X17_train, prep.y17_train, prep.X17_test, prep.y17_test),
                ]
            cases += [
                ("2018", "2017", prep.X18_train, prep.y18_train, prep.X17_all, prep.y17_all),
                ("2017", "2018", prep.X17_train, prep.y17_train, prep.X18_test, prep.y18_test),
            ]
            for src, tgt, Xtr, ytr, Xte, yte in cases:
                tag = f"cnn_{src}_to_{tgt}"
                m, pred = train_eval_hybrid(run, tag, Xtr, ytr, Xte, yte, cfg)
                record("CNN-BiLSTM-Attention", src, tgt, m, pred, yte)

        if args.only in {"all", "dann"}:
            # Exact notebook target pools and evaluation sets.
            m, pred = train_eval_dann(
                run, "dann_2018_to_2017",
                prep.X18_train, prep.y18_train,
                prep.X17_all, prep.y17_all,
                prep.X17_all, prep.y17_all, cfg,
            )
            record("Legacy-DANN-supervised-target", "2018", "2017", m, pred, prep.y17_all,
                   notes="Target class labels used during training; legacy reproduction only")

            m, pred = train_eval_dann(
                run, "dann_2017_to_2018",
                prep.X17_train, prep.y17_train,
                prep.X18_all, prep.y18_all,
                prep.X18_test, prep.y18_test, cfg,
            )
            record("Legacy-DANN-supervised-target", "2017", "2018", m, pred, prep.y18_test,
                   notes="Target class labels used during training; legacy reproduction only")

        table = pd.DataFrame(rows)
        run.save_dataframe("table.csv", table)
        if predictions:
            run.save_dataframe("predictions.csv", pd.concat(predictions, ignore_index=True))
        print("\n", table.to_string(index=False))
        print("\nExact feature order saved at:", run.run_dir / "feature_order_common.txt")
        print("Run artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
