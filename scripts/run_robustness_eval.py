#!/usr/bin/env python
"""Final E6: constraint-aware malicious-only FGSM/PGD robustness evaluation."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from ids.data.loaders import load_csv
from ids.data.canonical_mapping import resolve_mapping, align_frames_with_mapping
from ids.preprocessing.in_domain import preprocess_clean_binary_in_domain
from ids.preprocessing.cross_domain import prepare_clean_source_target, prepare_clean_source_target_fixed_test
from ids.models.hybrid import build_legacy_binary_hybrid
from ids.models.clean_dann import train_clean_dann
from ids.models.tabular import build_mlp_bn_fair_binary, shared_layer_weights
from ids.evaluation.metrics import binary, binary_confusion
from ids.evaluation.thresholds import choose_binary_threshold
from ids.evaluation.unsupervised_thresholds import choose_unsupervised_target_threshold
from ids.robustness.attacks import fgsm, pgd
from ids.robustness.constraints import build_semantic_projector
from ids.experiments.tracker import ExperimentTracker
from ids.training.imbalance import balanced_class_weights, class_weight_vector
from ids.utils.seed import set_global_seed
from run_transfer_clean import train_source_only


FINAL_VARIANTS = ("cnn", "mlp", "mlp_dann", "mlp_at", "mlp_dann_at")


def _class_probabilities(model, x):
    out = model.predict(x, verbose=0)
    return np.asarray(out[0] if isinstance(out, (list, tuple)) else out)


def eval_score(y, score, threshold):
    pred = (np.asarray(score) >= float(threshold)).astype(int)
    return {**binary(y, pred, score), **binary_confusion(y, pred)}


def _threshold_for_target(model, split, strategy: str, seed: int) -> float:
    source_val = _class_probabilities(model, split.X_source_val)[:, 1]
    source_threshold, _ = choose_binary_threshold(split.y_source_val, source_val, strategy="source_val_macro_f1")
    target_adapt = _class_probabilities(model, split.X_target_unlabeled)[:, 1]
    result = choose_unsupervised_target_threshold(
        strategy,
        target_adapt_score=target_adapt,
        source_threshold=source_threshold,
        source_positive_prevalence=float(np.mean(split.y_source_train)),
        seed=seed,
    )
    return float(result.threshold)


def _attack_rows(run, model, x, y, threshold, model_name, scenario, eps_values,
                 projector, constraint_cfg, input_kind, seed):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    malicious = np.flatnonzero(y == 1)
    if len(malicious) == 0:
        raise ValueError("E6 malicious-only evaluation received no malicious examples")
    clean_audit = projector.assert_clean_feasible(x[malicious])
    clean_inconsistencies = int(
        clean_audit.get("clean_nonnegative_violation_count", 0) +
        clean_audit.get("clean_ordered_triplet_violation_count", 0)
    )
    if clean_inconsistencies:
        print(
            f"WARNING: {clean_inconsistencies} pre-existing semantic inconsistencies "
            "were found in clean malicious rows. Their affected coordinates are "
            "frozen; only attack-induced violations are blocking."
        )

    clean_score = _class_probabilities(model, x)[:, 1]
    clean_metrics = eval_score(y, clean_score, threshold)
    clean_pred = (clean_score >= float(threshold)).astype(np.int32)
    initially_detected = malicious[clean_pred[malicious] == 1]
    rows = []

    for attack in ("fgsm", "pgd"):
        for eps in eps_values:
            eps = float(eps)
            xa = x.copy()
            if eps > 0:
                xm, ym = x[malicious], y[malicious]
                if attack == "fgsm":
                    adv = fgsm(model, xm, ym, epsilon=eps, project_fn=projector.project)
                else:
                    pgd_cfg = constraint_cfg["pgd"]
                    adv = pgd(
                        model, xm, ym, epsilon=eps, alpha=eps / 4.0,
                        steps=int(pgd_cfg["steps"]),
                        random_start=bool(pgd_cfg.get("random_start", True)),
                        seed=int(seed), project_fn=projector.project,
                    )
                xa[malicious] = adv

            audit = projector.audit(xa[malicious], x[malicious], eps)
            if not audit["constraint_pass"]:
                raise RuntimeError(f"Constraint gate failed for {model_name}/{attack}/eps={eps}: {audit}")
            robust_score = _class_probabilities(model, xa)[:, 1]
            robust_metrics = eval_score(y, robust_score, threshold)
            robust_pred = (robust_score >= float(threshold)).astype(np.int32)
            asr = float(np.mean(robust_pred[initially_detected] == 0)) if len(initially_detected) else float("nan")
            row = {
                "paper_stage": "E6", "paper_table": "T6_Robustness",
                "paper_figure": "F4_AttackEpsilon", "scenario": scenario,
                "model": model_name, "attack": attack, "epsilon": eps,
                "attack_scope": "malicious_only", "attack_space": "normalized_model_input",
                "constraint_protocol": projector.protocol,
                "constraint_status": (
                    "PASS_WITH_PREEXISTING_CLEAN_INCONSISTENCIES"
                    if clean_inconsistencies else "PASS"
                ),
                "target_labels_used_for_training": False,
                "target_labels_used_for_attack_generation": True,
                "decision_threshold": float(threshold), "input_kind": input_kind,
                "clean_macro_f1": float(clean_metrics["macro_f1"]),
                "robust_macro_f1": float(robust_metrics["macro_f1"]),
                "asr": asr, "attack_success_rate": asr,
                "fnr_adv": float(robust_metrics["fnr"]),
                "clean_fnr": float(clean_metrics["fnr"]),
                **robust_metrics, **audit,
            }
            rows.append(row)
            run.record_metrics(row)
    return rows


def train_in_domain_cnn(
    df, name, seed, epochs, patience, batch_size=256,
    checkpoint_path: Path | None = None, force_retrain: bool = False,
):
    split = preprocess_clean_binary_in_domain(df, name, seed=seed)
    model = build_legacy_binary_hybrid(split.X_train.shape[1], 1, lr=1e-3)
    weights = balanced_class_weights(split.y_train)
    sw = class_weight_vector(split.y_train, weights)

    class Monitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            p = self.model.predict(split.X_val[..., None], verbose=0)[:, 1]
            _threshold, macro = choose_binary_threshold(split.y_val, p, strategy="source_val_macro_f1")
            (logs if logs is not None else {})["val_macro_f1_opt"] = macro

    reused = False
    if checkpoint_path is not None and checkpoint_path.exists() and not force_retrain:
        try:
            model.load_weights(checkpoint_path)
            reused = True
            print(f"Reusing completed E6 CNN checkpoint: {checkpoint_path}")
        except Exception as exc:
            print(f"WARNING: checkpoint could not be loaded; retraining ({exc!r})")

    if not reused:
        callbacks = [Monitor()]
        best_checkpoint_path = None
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            best_checkpoint_path = checkpoint_path.with_name(
                checkpoint_path.name.replace(".weights.h5", ".best.weights.h5")
            )
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_checkpoint_path), monitor="val_macro_f1_opt", mode="max",
                save_best_only=True, save_weights_only=True, verbose=1,
            ))
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1_opt", mode="max", patience=patience, restore_best_weights=True
        ))
        model.fit(
            split.X_train[..., None], to_categorical(split.y_train, 2), sample_weight=sw,
            validation_data=(split.X_val[..., None], to_categorical(split.y_val, 2)),
            epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks,
        )
        # Only the final filename is auto-reused. The per-epoch ".best" file is
        # deliberately not treated as completed if training is interrupted.
        if checkpoint_path is not None:
            if best_checkpoint_path is not None and best_checkpoint_path.exists():
                model.load_weights(best_checkpoint_path)
            model.save_weights(checkpoint_path)
    val_score = model.predict(split.X_val[..., None], verbose=0)[:, 1]
    threshold, _ = choose_binary_threshold(split.y_val, val_score, strategy="source_val_macro_f1")
    return model, split, float(threshold)


def train_source_only_at(run, split, projector, *, epochs, batch_size, lr, patience,
                         epsilon, adversarial_fraction, seed, target_threshold_strategy):
    """Source-only MLP adversarial training; target labels are never inspected."""
    model = build_mlp_bn_fair_binary(split.X_source_train.shape[1], lr=lr)
    weights = balanced_class_weights(split.y_source_train)
    sw_all = class_weight_vector(split.y_source_train, weights)
    rng = np.random.default_rng(seed)
    best_weights, best_macro, bad = None, -np.inf, 0
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(split.X_source_train))
        losses = []
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = np.asarray(split.X_source_train[idx], dtype=np.float32).copy()
            yb = np.asarray(split.y_source_train[idx], dtype=np.int32)
            eligible = np.flatnonzero(yb == 1)
            n_adv = int(np.floor(len(eligible) * float(adversarial_fraction)))
            if n_adv > 0:
                chosen = rng.choice(eligible, size=n_adv, replace=False)
                xb[chosen] = fgsm(model, xb[chosen], yb[chosen], epsilon=float(epsilon), project_fn=projector.project)
            result = model.train_on_batch(
                xb, to_categorical(yb, 2), sample_weight=sw_all[idx], return_dict=True
            )
            losses.append(float(result["loss"]))

        val_score = _class_probabilities(model, split.X_source_val)[:, 1]
        _threshold, val_macro = choose_binary_threshold(split.y_source_val, val_score, strategy="source_val_macro_f1")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_f1": float(val_macro)})
        print(f"AT epoch={epoch:03d} loss={history[-1]['loss']:.4f} val_macro_f1={val_macro:.4f}")
        if val_macro > best_macro + 1e-6:
            best_macro, best_weights, bad = float(val_macro), model.get_weights(), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_weights is not None:
        model.set_weights(best_weights)
    run.save_dataframe("history_mlp_at.csv", pd.DataFrame(history))
    threshold = _threshold_for_target(model, split, target_threshold_strategy, seed)
    return model, threshold


def _prepare_cross_domain(args):
    source_raw, target_raw = load_csv(args.source_csv), load_csv(args.target_csv)
    mapping = resolve_mapping(
        source_raw.columns, target_raw.columns, args.source_name, args.target_name,
        mapping_path=args.mapping, verified_only=not args.allow_unverified_mapping,
    )
    source, target = align_frames_with_mapping(source_raw, target_raw, mapping)
    if args.target_test_csv:
        _, target_test = align_frames_with_mapping(source_raw, load_csv(args.target_test_csv), mapping)
        return prepare_clean_source_target_fixed_test(
            source, target, target_test, args.source_name, args.target_name,
            seed=args.seed, threshold=0.01, feature_alignment="exact",
        )
    return prepare_clean_source_target(
        source, target, args.source_name, args.target_name,
        seed=args.seed, threshold=0.01, feature_alignment="exact",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["in_domain_2018", "2018_to_CICIoT2023"], required=True)
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--source-name", default="2018")
    ap.add_argument("--target-csv")
    ap.add_argument("--target-test-csv")
    ap.add_argument("--target-name", default="CICIoT2023")
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--constraint-config", default="configs/robustness/e6_constraints_v051.yaml")
    ap.add_argument("--allow-unverified-mapping", action="store_true")
    ap.add_argument("--lambda-d", type=float, default=0.01)
    ap.add_argument("--eps", nargs="+", type=float, default=[0, 0.01, 0.03, 0.05, 0.1])
    ap.add_argument("--variants", nargs="+", choices=FINAL_VARIANTS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--checkpoint-root", default="results/checkpoints/e6_v052")
    ap.add_argument("--force-retrain", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-root", default="results")
    args = ap.parse_args()
    set_global_seed(args.seed)
    if args.smoke:
        args.epochs, args.patience = min(args.epochs, 5), min(args.patience, 2)

    if args.variants is None:
        args.variants = ["cnn"] if args.scenario == "in_domain_2018" else ["mlp", "mlp_dann", "mlp_at", "mlp_dann_at"]
    invalid = set(args.variants) - {"cnn"} if args.scenario == "in_domain_2018" else set(args.variants) & {"cnn"}
    if invalid:
        raise SystemExit(f"Invalid variants for {args.scenario}: {sorted(invalid)}")

    protocol = "v0.5.2_E6_constraint_aware_dirty_baseline"
    with ExperimentTracker(
        experiment="paper_E6_robustness", protocol=protocol, seed=args.seed,
        config=vars(args), output_root=args.output_root,
    ) as run:
        rows = []
        if args.scenario == "in_domain_2018":
            checkpoint = Path(args.checkpoint_root) / (
                f"cnn_in_domain_2018_{'smoke' if args.smoke else 'final'}_s{args.seed}.weights.h5"
            )
            model, split, threshold = train_in_domain_cnn(
                load_csv(args.source_csv), args.source_name, args.seed,
                args.epochs, args.patience, args.batch_size,
                checkpoint_path=checkpoint, force_retrain=args.force_retrain,
            )
            projector, constraint_cfg = build_semantic_projector(
                selected_feature_names=split.selected_feature_names, scaler=split.scaler,
                source_train_scaled=split.X_train, dataset_name=args.source_name,
                constraint_path=args.constraint_config, canonical_mapping_path=args.mapping,
            )
            rows += _attack_rows(
                run, model, split.X_test[..., None], split.y_test, threshold,
                "CNN-BiLSTM-Attention", args.scenario, args.eps, projector,
                constraint_cfg, "cnn_sequence_axis", args.seed,
            )
        else:
            if not args.target_csv:
                raise SystemExit("--target-csv is required for cross-domain robustness")
            split = _prepare_cross_domain(args)
            projector, constraint_cfg = build_semantic_projector(
                selected_feature_names=split.selected_feature_names, scaler=split.scaler,
                source_train_scaled=split.X_source_train, dataset_name=args.source_name,
                constraint_path=args.constraint_config, canonical_mapping_path=args.mapping,
            )
            at_cfg = constraint_cfg["adversarial_training"]
            clean_model = clean_threshold = at_model = at_threshold = None

            if any(v in args.variants for v in ("mlp", "mlp_dann")):
                clean_model, _hist, _weights, threshold_result, _metrics = train_source_only(
                    run, split, args.epochs, args.batch_size, 1e-3, args.patience,
                    "gmm_logit", args.seed, "mlp_bn_fair",
                )
                clean_threshold = float(threshold_result.threshold)
            if "mlp" in args.variants:
                rows += _attack_rows(
                    run, clean_model, split.X_target_test, split.y_target_test, clean_threshold,
                    "Canonical MLP", args.scenario, args.eps, projector,
                    constraint_cfg, "canonical_tabular", args.seed,
                )

            if "mlp_dann" in args.variants:
                result = train_clean_dann(
                    split.X_source_train, split.y_source_train, split.X_source_val,
                    split.y_source_val, split.X_target_unlabeled,
                    epochs=args.epochs, batch_size=args.batch_size, lr=1e-3,
                    lambda_d=args.lambda_d, domain_loss_weight=0.5,
                    class_weight_mode="balanced", patience=args.patience,
                    seed=args.seed, verbose=1, encoder="mlp_bn_fair",
                    initial_shared_weights=shared_layer_weights(clean_model), freeze_batchnorm=True,
                )
                threshold = _threshold_for_target(result.model, split, "gmm_logit", args.seed)
                rows += _attack_rows(
                    run, result.model, split.X_target_test, split.y_target_test, threshold,
                    "Canonical MLP+DANN", args.scenario, args.eps, projector,
                    constraint_cfg, "canonical_tabular", args.seed,
                )

            if any(v in args.variants for v in ("mlp_at", "mlp_dann_at")):
                at_model, at_threshold = train_source_only_at(
                    run, split, projector, epochs=args.epochs, batch_size=args.batch_size,
                    lr=1e-3, patience=args.patience,
                    epsilon=float(at_cfg["epsilon_train"]),
                    adversarial_fraction=float(at_cfg["adversarial_fraction"]),
                    seed=args.seed, target_threshold_strategy="gmm_logit",
                )
            if "mlp_at" in args.variants:
                rows += _attack_rows(
                    run, at_model, split.X_target_test, split.y_target_test, at_threshold,
                    "Canonical MLP+AT", args.scenario, args.eps, projector,
                    constraint_cfg, "canonical_tabular", args.seed,
                )

            if "mlp_dann_at" in args.variants:
                result_at = train_clean_dann(
                    split.X_source_train, split.y_source_train, split.X_source_val,
                    split.y_source_val, split.X_target_unlabeled,
                    epochs=args.epochs, batch_size=args.batch_size, lr=1e-3,
                    lambda_d=float(at_cfg.get("dann_lambda_d", args.lambda_d)),
                    domain_loss_weight=0.5, class_weight_mode="balanced",
                    patience=args.patience, seed=args.seed, verbose=1,
                    encoder="mlp_bn_fair", initial_shared_weights=shared_layer_weights(at_model),
                    freeze_batchnorm=True, adversarial_epsilon=float(at_cfg["epsilon_train"]),
                    adversarial_fraction=float(at_cfg["adversarial_fraction"]),
                    adversarial_project_fn=projector.project,
                )
                threshold = _threshold_for_target(result_at.model, split, "gmm_logit", args.seed)
                rows += _attack_rows(
                    run, result_at.model, split.X_target_test, split.y_target_test, threshold,
                    "Canonical MLP+DANN+AT", args.scenario, args.eps, projector,
                    constraint_cfg, "canonical_tabular", args.seed,
                )

        out = pd.DataFrame(rows)
        run.save_dataframe("table_robustness.csv", out)
        print("\n", out.to_string(index=False))
        print("\nE6 constraint gate: PASS for every emitted row.")
        print("Run artefacts:", run.run_dir)


if __name__ == "__main__":
    main()
