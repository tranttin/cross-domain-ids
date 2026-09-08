#!/usr/bin/env python
"""Leakage-resistant cross-domain baseline + DANN for new research.

Clean protocol guarantees
-------------------------
- deterministic feature order;
- selector/scaler fit on SOURCE TRAIN only;
- target class labels never enter training or model selection;
- binary class-imbalance correction uses SOURCE TRAIN labels only;
- checkpoint selection uses SOURCE VALIDATION Macro-F1;
- optional decision-threshold selection uses SOURCE VALIDATION only;
- every row logs prevalence, prediction rate, threshold, lambda and source weights.

Examples
--------
# Smoke-test only DANN in the missing direction after a reset:
python scripts/run_cross_domain_clean.py --direction 2018-2017 --only dann

# Try weaker domain reversal without editing YAML:
python scripts/run_cross_domain_clean.py --direction 2017-2018 --only dann --lambda-d 0.1

# Skip jobs already present in results/summary.csv with the same protocol/settings:
python scripts/run_cross_domain_clean.py --resume
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical

from ids.data.loaders import load_csv
from ids.preprocessing.cross_domain import prepare_clean_source_target
from ids.models.hybrid import build_legacy_binary_hybrid
from ids.models.clean_dann import train_clean_dann
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.thresholds import choose_binary_threshold
from ids.experiments.tracker import ExperimentTracker
from ids.training.imbalance import balanced_class_weights, class_weight_vector
from ids.utils.config import load_yaml
from ids.utils.seed import set_global_seed


def metrics_from_prob(y, prob, threshold: float):
    score = np.asarray(prob)[:, 1]
    pred = (score >= float(threshold)).astype(int)
    return {**binary(y, pred, score), **binary_confusion(y, pred)}, pred


class _SourceValMacroF1(tf.keras.callbacks.Callback):
    """Threshold-aware source-validation monitor for clean checkpointing."""

    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val[..., None]
        self.y_val = np.asarray(y_val).astype(int)

    def on_epoch_end(self, epoch, logs=None):
        from sklearn.metrics import balanced_accuracy_score
        logs = logs if logs is not None else {}
        prob = self.model.predict(self.X_val, verbose=0)
        thr, macro = choose_binary_threshold(
            self.y_val, prob[:, 1], strategy="source_val_macro_f1"
        )
        pred = (prob[:, 1] >= thr).astype(int)
        logs["val_macro_f1"] = float(macro)
        logs["val_balanced_accuracy"] = float(balanced_accuracy_score(self.y_val, pred))
        logs["val_decision_threshold"] = float(thr)


def _source_weights(y, mode: str) -> dict[int, float]:
    if mode == "balanced":
        return balanced_class_weights(y)
    if mode == "none":
        return {int(c): 1.0 for c in np.unique(y)}
    raise ValueError("class_weight_mode must be balanced or none")


def train_source_only(run, split, cfg, tag, class_weight_mode: str, threshold_strategy: str):
    deep = cfg["deep"]
    model = build_legacy_binary_hybrid(
        split.X_source_train.shape[1], 1,
        lr=float(deep.get("learning_rate", 1e-3)),
    )
    weights = _source_weights(split.y_source_train, class_weight_mode)
    sample_weight = class_weight_vector(split.y_source_train, weights)

    val_macro_cb = _SourceValMacroF1(split.X_source_val, split.y_source_val)
    callbacks = [
        val_macro_cb,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max",
            patience=int(deep.get("patience", 5)), restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1", mode="max",
            factor=0.3, patience=2, min_lr=1e-6,
        ),
        tf.keras.callbacks.CSVLogger(str(run.run_dir / f"history_{tag}.csv")),
    ]
    t0 = time.time()
    history = model.fit(
        split.X_source_train[..., None], to_categorical(split.y_source_train, 2),
        sample_weight=sample_weight,
        validation_data=(
            split.X_source_val[..., None],
            to_categorical(split.y_source_val, 2),
        ),
        epochs=int(deep.get("epochs", 30)),
        batch_size=int(deep.get("batch_size", 256)),
        shuffle=True, callbacks=callbacks, verbose=1,
    )

    val_prob = model.predict(split.X_source_val[..., None], verbose=0)
    threshold, threshold_val_metric = choose_binary_threshold(
        split.y_source_val, val_prob[:, 1], strategy=threshold_strategy
    )
    prob = model.predict(split.X_target_test[..., None], verbose=0)
    m, pred = metrics_from_prob(split.y_target_test, prob, threshold)
    m["train_seconds"] = round(time.time() - t0, 3)
    macro_hist = history.history.get("val_macro_f1", [])
    m["best_epoch"] = int(np.argmax(macro_hist) + 1) if macro_hist else None
    m["best_source_val_macro_f1"] = float(np.max(macro_hist)) if macro_hist else None
    m["decision_threshold"] = float(threshold)
    m["threshold_source_val_metric"] = float(threshold_val_metric)
    return m, pred, weights


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _already_done(
    summary: pd.DataFrame,
    *,
    protocol: str,
    seed: int,
    model: str,
    src: str,
    tgt: str,
    class_weight_mode: str,
    threshold_strategy: str,
    lambda_d: float | None,
    domain_loss_weight: float | None,
) -> bool:
    if summary.empty:
        return False
    required = {"protocol", "seed", "model", "train_domain", "test_domain"}
    if not required.issubset(summary.columns):
        return False
    mask = (
        (summary["protocol"].astype(str) == str(protocol))
        & (pd.to_numeric(summary["seed"], errors="coerce") == int(seed))
        & (summary["model"].astype(str) == model)
        & (summary["train_domain"].astype(str) == src)
        & (summary["test_domain"].astype(str) == tgt)
    )
    for col, value in [
        ("class_weight_mode", class_weight_mode),
        ("threshold_strategy", threshold_strategy),
    ]:
        if col not in summary.columns:
            return False
        mask &= summary[col].astype(str) == str(value)
    if lambda_d is not None:
        if "lambda_d" not in summary.columns:
            return False
        mask &= np.isclose(pd.to_numeric(summary["lambda_d"], errors="coerce"), float(lambda_d), equal_nan=False)
    if domain_loss_weight is not None:
        if "domain_loss_weight" not in summary.columns:
            return False
        mask &= np.isclose(pd.to_numeric(summary["domain_loss_weight"], errors="coerce"), float(domain_loss_weight), equal_nan=False)
    return bool(mask.any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/clean_cross_domain.yaml")
    ap.add_argument("--direction", choices=["both", "2017-2018", "2018-2017"], default="both")
    ap.add_argument("--only", choices=["all", "source_only", "dann"], default="all")
    ap.add_argument("--output-root", default="results")
    ap.add_argument("--seed", type=int, help="Override config seed.")
    ap.add_argument("--lambda-d", type=float, help="Override DANN GRL strength.")
    ap.add_argument("--domain-loss-weight", type=float, help="Override DANN domain-loss multiplier.")
    ap.add_argument("--class-weight", choices=["balanced", "none"], help="Override SOURCE-only class weighting.")
    ap.add_argument(
        "--threshold-strategy",
        choices=["fixed_0.5", "source_val_macro_f1", "source_val_balanced_accuracy"],
        help="Choose threshold using SOURCE validation only.",
    )
    ap.add_argument("--resume", action="store_true", help="Skip matching completed jobs in results/summary.csv.")
    ap.add_argument("--smoke", action="store_true", help="Engineering-only: cap deep epochs at 5 for fast collapse checks.")
    ap.add_argument("--max-epochs", type=int, help="Hard cap epochs without editing YAML.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    deep = cfg.setdefault("deep", {})
    evaluation = cfg.setdefault("evaluation", {})
    if args.lambda_d is not None:
        deep["lambda_d"] = float(args.lambda_d)
    if args.domain_loss_weight is not None:
        deep["domain_loss_weight"] = float(args.domain_loss_weight)
    if args.class_weight is not None:
        deep["class_weight_mode"] = args.class_weight
    if args.threshold_strategy is not None:
        evaluation["threshold_strategy"] = args.threshold_strategy

    class_weight_mode = str(deep.get("class_weight_mode", "balanced"))
    threshold_strategy = str(evaluation.get("threshold_strategy", "source_val_macro_f1"))
    lambda_d = float(deep.get("lambda_d", 0.1))
    domain_loss_weight = float(deep.get("domain_loss_weight", 0.5))
    protocol = "v0.5.0_clean_CIC_cross_domain"
    if args.max_epochs is not None:
        deep["epochs"] = min(int(deep.get("epochs", 30)), int(args.max_epochs))
    if args.smoke:
        deep["epochs"] = min(int(deep.get("epochs", 30)), 5)
        deep["patience"] = min(int(deep.get("patience", 5)), 3)
        protocol = protocol + "_smoke"

    set_global_seed(seed)
    df18 = load_csv(cfg["data"]["cic2018_csv"])
    df17 = load_csv(cfg["data"]["cic2017_csv"])
    directions = []
    if args.direction in {"both", "2017-2018"}:
        directions.append(("2017", df17, "2018", df18))
    if args.direction in {"both", "2018-2017"}:
        directions.append(("2018", df18, "2017", df17))

    summary = _load_summary(Path(args.output_root) / "summary.csv") if args.resume else pd.DataFrame()

    with ExperimentTracker(
        experiment="cross_domain_clean", protocol=protocol,
        seed=seed, config=cfg, output_root=args.output_root,
    ) as run:
        rows = []
        run.logger.info(
            "clean settings class_weight=%s threshold=%s lambda_d=%.4f domain_loss_weight=%.4f",
            class_weight_mode, threshold_strategy, lambda_d, domain_loss_weight,
        )

        for src_name, src_df, tgt_name, tgt_df in directions:
            run.logger.info("prepare clean direction %s->%s", src_name, tgt_name)
            split = prepare_clean_source_target(
                src_df, tgt_df, source_name=src_name, target_name=tgt_name,
                test_size=float(cfg["data"].get("test_size", 0.2)),
                validation_size=float(cfg["data"].get("validation_size", 0.1)),
                seed=seed,
                threshold=float(cfg["preprocessing"].get("variance_threshold", 0.01)),
                quantile_range=tuple(cfg["preprocessing"].get("robust_quantile_range", [5, 95])),
                feature_alignment=cfg["preprocessing"].get("feature_alignment", "exact"),
            )
            run.save_feature_names(split.selected_feature_names, f"features_{src_name}_to_{tgt_name}.json")
            run.logger.info(
                "%s->%s source train/val/test=%s/%s/%s target adapt/test=%s/%s",
                src_name, tgt_name, split.X_source_train.shape, split.X_source_val.shape,
                split.X_source_test.shape, split.X_target_unlabeled.shape, split.X_target_test.shape,
            )
            run.logger.info(
                "%s source attack prevalence train=%.4f val=%.4f | target-test prevalence=%.4f (evaluation only)",
                src_name,
                float(np.mean(split.y_source_train)), float(np.mean(split.y_source_val)),
                float(np.mean(split.y_target_test)),
            )

            if args.only in {"all", "source_only"}:
                if args.resume and _already_done(
                    summary, protocol=protocol, seed=seed,
                    model="CNN-BiLSTM-Attention", src=src_name, tgt=tgt_name,
                    class_weight_mode=class_weight_mode, threshold_strategy=threshold_strategy,
                    lambda_d=None, domain_loss_weight=None,
                ):
                    run.logger.info("SKIP completed source-only %s->%s", src_name, tgt_name)
                else:
                    m, pred, weights = train_source_only(
                        run, split, cfg, f"source_only_{src_name}_to_{tgt_name}",
                        class_weight_mode, threshold_strategy,
                    )
                    row = {
                        "model": "CNN-BiLSTM-Attention", "train_domain": src_name,
                        "test_domain": tgt_name, "task": "binary", "adaptation": "none",
                        "target_labels_in_training": False,
                        "class_weight_mode": class_weight_mode,
                        "source_weight_0": weights.get(0), "source_weight_1": weights.get(1),
                        "threshold_strategy": threshold_strategy,
                        "lambda_d": np.nan, "domain_loss_weight": np.nan,
                        **m,
                    }
                    rows.append(row)
                    run.record_metrics(row)

            if args.only in {"all", "dann"}:
                if args.resume and _already_done(
                    summary, protocol=protocol, seed=seed,
                    model="Clean-DANN", src=src_name, tgt=tgt_name,
                    class_weight_mode=class_weight_mode, threshold_strategy=threshold_strategy,
                    lambda_d=lambda_d, domain_loss_weight=domain_loss_weight,
                ):
                    run.logger.info("SKIP completed Clean-DANN %s->%s", src_name, tgt_name)
                else:
                    t0 = time.time()
                    result = train_clean_dann(
                        split.X_source_train, split.y_source_train,
                        split.X_source_val, split.y_source_val,
                        split.X_target_unlabeled,
                        epochs=int(deep.get("epochs", 30)),
                        batch_size=int(deep.get("batch_size", 256)),
                        lr=float(deep.get("dann_learning_rate", 1e-4)),
                        lambda_d=lambda_d,
                        domain_loss_weight=domain_loss_weight,
                        class_weight_mode=class_weight_mode,
                        patience=int(deep.get("patience", 5)), seed=seed, verbose=1,
                    )
                    run.save_history(result.history, f"history_clean_dann_{src_name}_to_{tgt_name}.csv")
                    run.write_json(
                        f"source_class_weights_dann_{src_name}_to_{tgt_name}.json",
                        result.source_class_weights or {},
                    )

                    val_cls, _ = result.model.predict(split.X_source_val[..., None], verbose=0)
                    decision_threshold, threshold_val_metric = choose_binary_threshold(
                        split.y_source_val, val_cls[:, 1], strategy=threshold_strategy
                    )
                    prob, _ = result.model.predict(split.X_target_test[..., None], verbose=0)
                    m, pred = metrics_from_prob(split.y_target_test, prob, decision_threshold)
                    m["train_seconds"] = round(time.time() - t0, 3)
                    m["best_epoch"] = result.best_epoch
                    m["best_source_val_macro_f1"] = result.best_val_macro_f1
                    m["decision_threshold"] = float(decision_threshold)
                    m["threshold_source_val_metric"] = float(threshold_val_metric)
                    weights = result.source_class_weights or {}
                    row = {
                        "model": "Clean-DANN", "train_domain": src_name,
                        "test_domain": tgt_name, "task": "binary", "adaptation": "UDA",
                        "target_labels_in_training": False,
                        "class_weight_mode": class_weight_mode,
                        "source_weight_0": weights.get(0), "source_weight_1": weights.get(1),
                        "threshold_strategy": threshold_strategy,
                        "lambda_d": lambda_d, "domain_loss_weight": domain_loss_weight,
                        "final_val_domain_accuracy": result.history.get("val_domain_accuracy", [np.nan])[-1],
                        "final_unlabelled_target_predicted_positive_rate": result.history.get("target_predicted_positive_rate", [np.nan])[-1],
                        **m,
                    }
                    rows.append(row)
                    run.record_metrics(row)

        table = pd.DataFrame(rows)
        run.save_dataframe("table.csv", table)
        if len(table):
            print("\n", table.to_string(index=False))
        else:
            print("\nNo new jobs were run (all matching jobs were skipped).")
        print("Run artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
