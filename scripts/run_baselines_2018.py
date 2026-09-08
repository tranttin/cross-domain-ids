#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import time
import pandas as pd

from ids.data.loaders import load_csv
from ids.preprocessing.in_domain import preprocess_legacy_2018
from ids.models.classical import build_legacy_multiclass_models
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
        experiment="baselines_2018",
        protocol="legacy_notebook",
        seed=seed,
        config=cfg,
        output_root=args.output_root,
    ) as run:
        df = load_csv(cfg["data"]["processed_csv"])
        split = preprocess_legacy_2018(
            df,
            test_size=cfg["data"].get("test_size", 0.2),
            seed=seed,
            variance_threshold=cfg["preprocessing"].get("variance_threshold", 0.01),
            quantile_range=tuple(cfg["preprocessing"].get("robust_quantile_range", [5, 95])),
        )
        run.logger.info("train=%s test=%s classes=%s", split.X_train_ml.shape, split.X_test_ml.shape, split.n_classes)
        run.save_feature_names(split.selected_feature_names)

        k = min(int(cfg.get("classical", {}).get("select_k_best", 30)), split.X_train_ml.shape[1])
        models = build_legacy_multiclass_models(k=k, seed=seed)
        rows = []
        for name, model in models.items():
            run.logger.info("training model=%s", name)
            t0 = time.time()
            model.fit(split.X_train_ml, split.y_train)
            pred = model.predict(split.X_test_ml)
            metrics = multiclass_weighted(split.y_test, pred)
            row = {
                "model": name,
                "train_domain": "2018",
                "test_domain": "2018",
                "task": "multiclass",
                "train_seconds": round(time.time() - t0, 3),
                **metrics,
            }
            rows.append(row)
            run.record_metrics(row)

        table = pd.DataFrame(rows)
        run.save_dataframe("table.csv", table)
        print("\n", table.to_string(index=False))
        print("\nRun artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
