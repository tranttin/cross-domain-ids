from __future__ import annotations

"""Score-distribution diagnostics for cross-domain binary classification.

The functions in this module separate three concepts that are easy to conflate in
cross-domain IDS experiments:

1. ranking quality (AUROC/AUPRC),
2. score calibration / location shift, and
3. the hard decision induced by a threshold selected on labelled SOURCE validation.

Target labels are optional. Whenever they are supplied, outputs are strictly
DIAGNOSTIC / ORACLE quantities and must never be used for UDA training, checkpoint
selection, or deployable threshold selection.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


DEFAULT_QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


@dataclass(frozen=True)
class ScoreMetadata:
    method: str
    classes: list[Any]
    positive_class_index: int | None


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def positive_class_score(model, X) -> tuple[np.ndarray, ScoreMetadata]:
    """Return a 0..1 score oriented toward class 1.

    For predict_proba estimators, the class-1 column is resolved from ``classes_``
    instead of assuming column 1. For binary decision_function estimators, sklearn's
    score is oriented toward ``classes_[1]``; the sign is flipped when class 1 is
    stored in position 0, then mapped monotonically through a sigmoid.
    """
    classes_arr = np.asarray(getattr(model, "classes_", [0, 1]))
    classes = [_json_scalar(v) for v in classes_arr.tolist()]

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        matches = np.where(classes_arr == 1)[0]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one positive class=1, got classes_={classes}")
        idx = int(matches[0])
        if proba.ndim != 2 or idx >= proba.shape[1]:
            raise ValueError(f"predict_proba shape {proba.shape} incompatible with classes_={classes}")
        return proba[:, idx], ScoreMetadata("predict_proba", classes, idx)

    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        matches = np.where(classes_arr == 1)[0]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one positive class=1, got classes_={classes}")
        idx = int(matches[0])
        if raw.ndim == 1:
            # sklearn binary decision_function is positive toward classes_[1].
            if len(classes_arr) == 2 and idx == 0:
                raw = -raw
        elif raw.ndim == 2:
            raw = raw[:, idx]
        else:
            raise ValueError(f"Unsupported decision_function shape: {raw.shape}")
        score = 1.0 / (1.0 + np.exp(-np.clip(raw, -50, 50)))
        return score, ScoreMetadata("decision_function_sigmoid", classes, idx)

    pred = np.asarray(model.predict(X)).astype(int)
    return pred.astype(float), ScoreMetadata("hard_prediction", classes, None)


def score_quantile_frame(
    source_val_score,
    target_score,
    *,
    source_val_labels=None,
    target_labels=None,
    quantiles=DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Long-form score quantiles for source validation and target evaluation.

    Target class-conditional rows are labelled ``diagnostic_target_label=True``.
    """
    rows: list[dict[str, Any]] = []

    def add(split: str, scores, labels, diagnostic_target_label: bool):
        scores = np.asarray(scores, dtype=float)
        groups = [("all", np.ones(len(scores), dtype=bool))]
        if labels is not None:
            y = np.asarray(labels).astype(int)
            groups.extend([("benign", y == 0), ("attack", y == 1)])
        for group, mask in groups:
            vals = scores[mask]
            if len(vals) == 0:
                continue
            qs = np.quantile(vals, quantiles)
            for q, value in zip(quantiles, qs):
                rows.append({
                    "split": split,
                    "group": group,
                    "quantile": float(q),
                    "score": float(value),
                    "n": int(len(vals)),
                    "diagnostic_target_label": bool(diagnostic_target_label and group != "all"),
                })

    add("source_val", source_val_score, source_val_labels, False)
    add("target", target_score, target_labels, target_labels is not None)
    return pd.DataFrame(rows)


def _median(scores, labels=None, cls=None) -> float:
    s = np.asarray(scores, dtype=float)
    if labels is not None and cls is not None:
        y = np.asarray(labels).astype(int)
        s = s[y == cls]
    return float(np.median(s)) if len(s) else float("nan")


def choose_target_oracle_threshold(y_true, score, grid_size: int = 401) -> dict[str, float]:
    """Target-labelled threshold upper bound for diagnostics ONLY.

    Candidate thresholds are derived from the target score distribution, so this can
    expose a pure threshold/location shift even when all target scores lie below the
    source-selected threshold. Never use this threshold in a clean UDA result.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {
            "threshold": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
        }

    q = np.linspace(0.0, 1.0, max(3, int(grid_size)))
    candidates = np.unique(np.quantile(s, q))
    if len(candidates):
        candidates = np.unique(np.concatenate([
            [np.nextafter(float(candidates[0]), -np.inf)],
            candidates,
            [np.nextafter(float(candidates[-1]), np.inf)],
        ]))

    best_t = 0.5
    best_f1 = -np.inf
    best_bal = float("nan")
    for t in candidates:
        pred = (s >= t).astype(int)
        macro = float(f1_score(y, pred, average="macro", zero_division=0))
        bal = float(balanced_accuracy_score(y, pred))
        if macro > best_f1 + 1e-12 or (
            abs(macro - best_f1) <= 1e-12 and abs(float(t) - 0.5) < abs(best_t - 0.5)
        ):
            best_t, best_f1, best_bal = float(t), macro, bal
    return {"threshold": best_t, "macro_f1": best_f1, "balanced_accuracy": best_bal}


def build_score_shift_diagnostic(
    *,
    source_val_score,
    source_val_labels,
    source_threshold: float,
    target_score,
    target_labels=None,
) -> dict[str, Any]:
    """Summarize ranking, score shift and threshold-transfer behavior."""
    sv = np.asarray(source_val_score, dtype=float)
    sy = np.asarray(source_val_labels).astype(int)
    ts = np.asarray(target_score, dtype=float)

    out: dict[str, Any] = {
        "source_threshold": float(source_threshold),
        "source_val_score_p01": float(np.quantile(sv, 0.01)),
        "source_val_score_median": _median(sv),
        "source_val_score_p99": float(np.quantile(sv, 0.99)),
        "source_val_benign_score_median": _median(sv, sy, 0),
        "source_val_attack_score_median": _median(sv, sy, 1),
        "target_score_p01": float(np.quantile(ts, 0.01)),
        "target_score_median": _median(ts),
        "target_score_p99": float(np.quantile(ts, 0.99)),
        "target_fraction_above_source_threshold": float(np.mean(ts >= source_threshold)),
        "target_score_location_ratio_to_threshold": (
            float(_median(ts) / source_threshold) if abs(source_threshold) > 1e-12 else float("nan")
        ),
        "uses_target_labels_for_selection": False,
    }

    if target_labels is not None:
        ty = np.asarray(target_labels).astype(int)
        out["diagnostic_target_labels_used"] = True
        out["diagnostic_target_benign_score_median"] = _median(ts, ty, 0)
        out["diagnostic_target_attack_score_median"] = _median(ts, ty, 1)
        out["diagnostic_target_class_median_gap"] = (
            out["diagnostic_target_attack_score_median"] - out["diagnostic_target_benign_score_median"]
        )
        if len(np.unique(ty)) == 2:
            raw_auc = float(roc_auc_score(ty, ts))
            rev_auc = float(roc_auc_score(ty, 1.0 - ts))
        else:
            raw_auc = rev_auc = float("nan")
        out["diagnostic_target_auroc_raw"] = raw_auc
        out["diagnostic_target_auroc_reversed"] = rev_auc
        out["diagnostic_orientation_suspect"] = bool(
            np.isfinite(raw_auc) and np.isfinite(rev_auc) and raw_auc <= 0.25 and rev_auc >= 0.75
        )
        oracle = choose_target_oracle_threshold(ty, ts)
        out["diagnostic_target_oracle_threshold"] = oracle["threshold"]
        out["diagnostic_target_oracle_macro_f1"] = oracle["macro_f1"]
        out["diagnostic_target_oracle_balanced_accuracy"] = oracle["balanced_accuracy"]
        source_pred = (ts >= source_threshold).astype(int)
        source_macro = float(f1_score(ty, source_pred, average="macro", zero_division=0))
        out["diagnostic_source_threshold_target_macro_f1"] = source_macro
        out["diagnostic_target_oracle_macro_f1_gain"] = oracle["macro_f1"] - source_macro
    else:
        out["diagnostic_target_labels_used"] = False

    # Score-location collapse can be detected without target labels.
    fraction = out["target_fraction_above_source_threshold"]
    out["unlabelled_threshold_collapse_suspect"] = bool(fraction <= 0.01 or fraction >= 0.99)
    return out
