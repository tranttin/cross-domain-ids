from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)


def multiclass_weighted(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def binary(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    prevalence = float(np.mean(y_true == 1)) if len(y_true) else float("nan")
    predicted_positive_rate = float(np.mean(y_pred == 1)) if len(y_pred) else float("nan")
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "positive_prevalence": prevalence,
        "predicted_positive_rate": predicted_positive_rate,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["auprc"] = average_precision_score(y_true, y_score)
            out["auprc_baseline"] = prevalence
            out["auprc_lift"] = out["auprc"] - prevalence
        except Exception:
            out["auprc"] = float("nan")
            out["auprc_baseline"] = prevalence
            out["auprc_lift"] = float("nan")
        try:
            out["auroc"] = roc_auc_score(y_true, y_score)
        except Exception:
            out["auroc"] = float("nan")
    return out


def binary_confusion(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "fpr": fpr, "fnr": fnr}
