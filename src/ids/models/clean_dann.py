from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ids.models.dann import build_legacy_dann
from ids.models.tabular import (build_mlp_dann, build_mlp_fair_dann, build_mlp_bn_fair_dann, load_shared_layer_weights)
from ids.training.imbalance import balanced_class_weights
from ids.evaluation.thresholds import choose_binary_threshold


@dataclass
class CleanDANNResult:
    model: object
    history: dict[str, list[float]]
    best_epoch: int | None = None
    best_val_macro_f1: float | None = None
    source_class_weights: dict[int, float] | None = None


def train_clean_dann(
    X_source_train: np.ndarray,
    y_source_train: np.ndarray,
    X_source_val: np.ndarray,
    y_source_val: np.ndarray,
    X_target_unlabeled: np.ndarray,
    *,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-4,
    lambda_d: float = 0.1,
    domain_loss_weight: float = 0.5,
    class_weight_mode: str = "balanced",
    patience: int = 5,
    seed: int = 42,
    verbose: int = 1,
    abort_on_collapse: bool = False,
    collapse_patience: int = 2,
    encoder: str = "cnn_bilstm_attention",
    initial_shared_weights: dict[str, list] | None = None,
    freeze_batchnorm: bool = False,
    adversarial_epsilon: float = 0.0,
    adversarial_fraction: float = 0.5,
    adversarial_project_fn=None,
) -> CleanDANNResult:
    """Train source-labelled / target-unlabelled DANN without target class labels.

    Important clean-protocol rules
    ------------------------------
    * Classification loss uses labelled SOURCE batches only.
    * Domain loss uses source + UNLABELLED target batches.
    * Class weights, when enabled, are computed from SOURCE TRAIN labels only.
    * Checkpoint selection/early stopping uses SOURCE VALIDATION Macro-F1 only.
    * Target labels are deliberately absent from this function.
    * ``encoder="mlp"`` is a v0.4.5 canonical-tabular architecture control;
      the default preserves the recovered CNN--BiLSTM--Attention DANN path.
    """
    import tensorflow as tf
    from sklearn.metrics import balanced_accuracy_score, f1_score

    tf.keras.utils.set_random_seed(seed)
    if encoder == "cnn_bilstm_attention":
        model = build_legacy_dann(
            timesteps=X_source_train.shape[1], channels=1,
            lambda_d=lambda_d, lr=lr, domain_loss_weight=domain_loss_weight,
        )
        def adapt_x(x):
            return np.asarray(x, dtype=np.float32)[..., None]
    elif encoder == "mlp":
        model = build_mlp_dann(
            input_dim=X_source_train.shape[1],
            lambda_d=lambda_d, lr=lr, domain_loss_weight=domain_loss_weight,
        )
        def adapt_x(x):
            return np.asarray(x, dtype=np.float32)
    elif encoder == "mlp_fair":
        model = build_mlp_fair_dann(
            input_dim=X_source_train.shape[1],
            lambda_d=lambda_d, lr=lr, domain_loss_weight=domain_loss_weight,
        )
        def adapt_x(x):
            return np.asarray(x, dtype=np.float32)
    elif encoder == "mlp_bn_fair":
        model = build_mlp_bn_fair_dann(
            input_dim=X_source_train.shape[1],
            lambda_d=lambda_d, lr=lr, domain_loss_weight=domain_loss_weight,
        )
        if initial_shared_weights is not None:
            load_shared_layer_weights(model, initial_shared_weights, freeze_batchnorm=freeze_batchnorm)
        elif freeze_batchnorm:
            model.get_layer("bnfair_bn_1").trainable = False
        def adapt_x(x):
            return np.asarray(x, dtype=np.float32)
    else:
        raise ValueError("encoder must be 'cnn_bilstm_attention', 'mlp', 'mlp_fair', or 'mlp_bn_fair'")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    cls_loss_each = tf.keras.losses.SparseCategoricalCrossentropy(reduction="none")
    dom_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    if class_weight_mode == "balanced":
        class_weights = balanced_class_weights(y_source_train)
    elif class_weight_mode in {"none", "off", "false"}:
        class_weights = {int(c): 1.0 for c in np.unique(y_source_train)}
    else:
        raise ValueError("class_weight_mode must be 'balanced' or 'none'")
    # Dense vector makes batch lookup cheap and deterministic for binary labels.
    max_class = max(class_weights) if class_weights else 1
    weight_lookup_np = np.ones(max_class + 1, dtype=np.float32)
    for c, w in class_weights.items():
        weight_lookup_np[int(c)] = float(w)
    weight_lookup = tf.constant(weight_lookup_np, dtype=tf.float32)

    src_ds = tf.data.Dataset.from_tensor_slices(
        (adapt_x(X_source_train), y_source_train.astype(np.int32))
    ).shuffle(len(X_source_train), seed=seed, reshuffle_each_iteration=True).batch(
        batch_size, drop_remainder=False
    )
    tgt_ds = tf.data.Dataset.from_tensor_slices(
        adapt_x(X_target_unlabeled)
    ).shuffle(len(X_target_unlabeled), seed=seed + 1, reshuffle_each_iteration=True).batch(
        batch_size, drop_remainder=False
    )
    tgt_iterable = tgt_ds.repeat()

    Xv = tf.convert_to_tensor(adapt_x(X_source_val))
    yv = tf.convert_to_tensor(y_source_val.astype(np.int32))
    # Fixed unlabeled target subset for safety diagnostics only. No target class label is used.
    monitor_n = min(4096, len(X_target_unlabeled))
    Xt_monitor = tf.convert_to_tensor(adapt_x(X_target_unlabeled[:monitor_n]))

    history = {
        "loss": [], "cls_loss": [], "dom_loss": [],
        "val_cls_loss": [], "val_accuracy": [],
        "val_macro_f1": [], "val_balanced_accuracy": [],
        "val_decision_threshold": [], "val_domain_accuracy": [],
        "val_domain_balanced_accuracy": [],
        "val_domain_source_accuracy": [],
        "val_domain_target_accuracy": [],
        "val_domain_reversed_balanced_accuracy": [],
        "val_domain_score_gap": [],
        "target_predicted_positive_rate": [],
        "target_predicted_positive_rate_at_source_threshold": [],
    }
    best_weights = None
    best_epoch = None
    best_macro_f1 = -np.inf
    bad = 0
    collapse_bad = 0

    if adversarial_epsilon < 0:
        raise ValueError("adversarial_epsilon must be non-negative")
    if not 0.0 <= adversarial_fraction <= 1.0:
        raise ValueError("adversarial_fraction must be in [0, 1]")
    if adversarial_epsilon > 0 and adversarial_project_fn is None:
        raise ValueError("Constraint-aware adversarial training requires adversarial_project_fn")

    for epoch in range(1, epochs + 1):
        epoch_total, epoch_cls, epoch_dom, steps = 0.0, 0.0, 0.0, 0
        tgt_it = iter(tgt_iterable)
        for xs, ys in src_ds:
            xt = next(tgt_it)
            xs_for_training = xs
            if adversarial_epsilon > 0 and adversarial_fraction > 0:
                # E6 is an evasion threat model: only labelled malicious SOURCE
                # examples are eligible.  Half of those examples are replaced by
                # constraint-projected FGSM examples; target labels are never used.
                from ids.robustness.attacks import fgsm

                positive = np.flatnonzero(ys.numpy() == 1)
                n_adv = int(np.floor(len(positive) * adversarial_fraction))
                if n_adv > 0:
                    chosen = positive[:n_adv]
                    xs_np = xs.numpy()
                    adv_subset = fgsm(
                        model,
                        xs_np[chosen],
                        ys.numpy()[chosen],
                        epsilon=float(adversarial_epsilon),
                        project_fn=adversarial_project_fn,
                    )
                    xs_np[chosen] = adv_subset
                    xs_for_training = tf.convert_to_tensor(xs_np, dtype=tf.float32)
            with tf.GradientTape() as tape:
                # v0.4.7: one mixed-domain forward pass. This avoids sequential
                # source/target normalization updates and makes the domain loss
                # operate on a single balanced mixed mini-batch. Target class
                # logits are computed but never used in the classification loss.
                n_source = tf.shape(xs_for_training)[0]
                x_joint = tf.concat([xs_for_training, xt], axis=0)
                cls_joint, dom_joint = model(x_joint, training=True)
                cls_s = cls_joint[:n_source]
                dom_s = dom_joint[:n_source]
                dom_t = dom_joint[n_source:]

                per_example_cls = cls_loss_each(ys, cls_s)
                sample_weights = tf.gather(weight_lookup, ys)
                cls_loss = tf.reduce_sum(per_example_cls * sample_weights) / tf.maximum(
                    tf.reduce_sum(sample_weights), tf.keras.backend.epsilon()
                )

                dom_labels_s = tf.zeros((tf.shape(xs_for_training)[0],), dtype=tf.int32)
                dom_labels_t = tf.ones((tf.shape(xt)[0],), dtype=tf.int32)
                dom_loss = 0.5 * (
                    dom_loss_fn(dom_labels_s, dom_s) + dom_loss_fn(dom_labels_t, dom_t)
                )
                total = cls_loss + domain_loss_weight * dom_loss
                if model.losses:
                    total = total + tf.add_n(model.losses)
            grads = tape.gradient(total, model.trainable_variables)
            grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
            optimizer.apply_gradients(grads_and_vars)
            epoch_total += float(total.numpy())
            epoch_cls += float(cls_loss.numpy())
            epoch_dom += float(dom_loss.numpy())
            steps += 1

        val_cls, val_dom_s = model(Xv, training=False)
        target_cls_monitor, val_dom_t = model(Xt_monitor, training=False)
        val_score = val_cls[:, 1].numpy()
        yv_np = y_source_val.astype(np.int32)
        # v0.4.4 hotfix: checkpointing is threshold-aware. Threshold selection uses
        # labelled SOURCE validation only. Fixed 0.5 caused false early stopping in
        # v0.4.3 after class weighting/calibration moved the operating point.
        val_threshold, val_macro = choose_binary_threshold(
            yv_np, val_score, strategy="source_val_macro_f1"
        )
        val_pred = (val_score >= val_threshold).astype(np.int32)
        val_loss = float(tf.reduce_mean(cls_loss_each(yv, val_cls)).numpy())
        val_acc = float(np.mean(val_pred == yv_np))
        val_bal = float(balanced_accuracy_score(yv_np, val_pred))
        source_dom_pred = np.argmax(val_dom_s.numpy(), axis=1)
        target_dom_pred = np.argmax(val_dom_t.numpy(), axis=1)
        domain_correct = int(np.sum(source_dom_pred == 0)) + int(np.sum(target_dom_pred == 1))
        domain_total = len(source_dom_pred) + len(target_dom_pred)
        val_domain_acc = float(domain_correct / max(domain_total, 1))
        source_domain_acc = float(np.mean(source_dom_pred == 0))
        target_domain_acc = float(np.mean(target_dom_pred == 1))
        val_domain_bal_acc = 0.5 * (source_domain_acc + target_domain_acc)
        val_domain_rev_bal_acc = 1.0 - val_domain_bal_acc
        source_p_target = val_dom_s.numpy()[:, 1]
        target_p_target = val_dom_t.numpy()[:, 1]
        val_domain_score_gap = float(np.mean(target_p_target) - np.mean(source_p_target))
        target_score_monitor = target_cls_monitor[:, 1].numpy()
        target_pred_rate_05 = float(np.mean(target_score_monitor >= 0.5))
        target_pred_rate_source_t = float(np.mean(target_score_monitor >= val_threshold))

        denom = max(steps, 1)
        history["loss"].append(epoch_total / denom)
        history["cls_loss"].append(epoch_cls / denom)
        history["dom_loss"].append(epoch_dom / denom)
        history["val_cls_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_macro_f1"].append(val_macro)
        history["val_balanced_accuracy"].append(val_bal)
        history["val_decision_threshold"].append(float(val_threshold))
        history["val_domain_accuracy"].append(val_domain_acc)
        history["val_domain_balanced_accuracy"].append(float(val_domain_bal_acc))
        history["val_domain_source_accuracy"].append(float(source_domain_acc))
        history["val_domain_target_accuracy"].append(float(target_domain_acc))
        history["val_domain_reversed_balanced_accuracy"].append(float(val_domain_rev_bal_acc))
        history["val_domain_score_gap"].append(float(val_domain_score_gap))
        history["target_predicted_positive_rate"].append(target_pred_rate_05)
        history["target_predicted_positive_rate_at_source_threshold"].append(target_pred_rate_source_t)

        if verbose:
            print(
                f"epoch={epoch:03d} loss={history['loss'][-1]:.4f} "
                f"cls={history['cls_loss'][-1]:.4f} dom={history['dom_loss'][-1]:.4f} "
                f"val_cls={val_loss:.4f} val_acc={val_acc:.4f} "
                f"val_macro_f1={val_macro:.4f} val_bal_acc={val_bal:.4f} "
                f"val_thr={val_threshold:.3f} dom_acc={val_domain_acc:.4f} "
                f"dom_bal={val_domain_bal_acc:.4f} dom_gap={val_domain_score_gap:+.4f} "
                f"target_pos@0.5={target_pred_rate_05:.4f} "
                f"target_pos@src_thr={target_pred_rate_source_t:.4f}"
            )

        # Do not stop on target score collapse by default. v0.4.3 diagnostics showed
        # that score/calibration shift can make a strong ranker look all-zero at 0.5.
        # Target rates are diagnostics only and target labels are never used here.
        if abort_on_collapse:
            collapsed = target_pred_rate_source_t <= 0.001 or target_pred_rate_source_t >= 0.999
            if epoch >= 3 and collapsed and val_macro < 0.55:
                collapse_bad += 1
            else:
                collapse_bad = 0
            if collapse_bad >= max(1, int(collapse_patience)):
                if verbose:
                    print(
                        f"Safety abort at epoch {epoch}: target predictions collapsed at "
                        f"source-val threshold (positive_rate={target_pred_rate_source_t:.4f}) "
                        f"while source-val Macro-F1={val_macro:.4f}."
                    )
                break

        if val_macro > best_macro_f1 + 1e-6:
            best_macro_f1 = val_macro
            best_epoch = epoch
            best_weights = model.get_weights()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best source-val Macro-F1={best_macro_f1:.6f} at epoch {best_epoch}"
                    )
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return CleanDANNResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_macro_f1=None if best_epoch is None else float(best_macro_f1),
        source_class_weights=class_weights,
    )
