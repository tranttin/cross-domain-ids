#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from ids.data.loaders import load_csv
from ids.preprocessing.in_domain import preprocess_legacy_2018
from ids.models.hybrid import build_legacy_multiclass_hybrid
from ids.evaluation.metrics import multiclass_weighted
from ids.experiments.tracker import ExperimentTracker
from ids.utils.config import load_yaml
from ids.utils.seed import set_global_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/legacy_2018.yaml")
    ap.add_argument("--output-root", default="results")
    ap.add_argument("--seed", type=int, help="Override config seed.")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_global_seed(seed)

    with ExperimentTracker(
        experiment="hybrid_2018",
        protocol="legacy_notebook_test_as_validation",
        seed=seed,
        config=cfg,
        output_root=args.output_root,
    ) as run:
        df = load_csv(cfg["data"]["processed_csv"])
        s = preprocess_legacy_2018(
            df,
            test_size=cfg["data"].get("test_size", 0.2), seed=seed,
            variance_threshold=cfg["preprocessing"].get("variance_threshold", 0.01),
            quantile_range=tuple(cfg["preprocessing"].get("robust_quantile_range", [5, 95])),
        )
        run.save_feature_names(s.selected_feature_names)
        ytr = to_categorical(s.y_train, num_classes=s.n_classes)
        yte = to_categorical(s.y_test, num_classes=s.n_classes)
        deep = cfg["deep"]
        model = build_legacy_multiclass_hybrid(
            s.X_train_cnn.shape[1], 1, s.n_classes,
            lr=float(deep.get("learning_rate", 1e-4)),
        )
        run.save_model_summary(model)
        train_ds = tf.data.Dataset.from_tensor_slices((s.X_train_cnn, ytr)).shuffle(
            len(s.X_train_cnn), seed=seed
        ).batch(int(deep.get("batch_size", 256))).prefetch(tf.data.AUTOTUNE)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=int(deep.get("early_stopping_patience", 6)), restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=int(deep.get("reduce_lr_patience", 3)), min_lr=1e-6
            ),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.CSVLogger(str(run.run_dir / "keras_epoch_log.csv")),
        ]
        t0 = time.time()
        # LEGACY ONLY: the notebook used test data as validation data for early stopping.
        history = model.fit(
            train_ds,
            epochs=int(deep.get("epochs", 50)),
            validation_data=(s.X_test_cnn, yte),
            callbacks=callbacks,
            verbose=1,
        )
        run.save_history(history)
        prob = model.predict(s.X_test_cnn, verbose=0)
        pred = prob.argmax(axis=1)
        metrics = multiclass_weighted(s.y_test, pred)
        row = {
            "model": "CNN-BiLSTM-Attention",
            "train_domain": "2018",
            "test_domain": "2018",
            "task": "multiclass",
            "train_seconds": round(time.time() - t0, 3),
            "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
            **metrics,
        }
        run.record_metrics(row)
        run.save_dataframe("predictions.csv", pd.DataFrame({"y_true": s.y_test, "y_pred": pred}))
        print("\n", row)
        print("Run artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
