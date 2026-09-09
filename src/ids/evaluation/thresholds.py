from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def choose_binary_threshold(
    y_true,
    positive_score,
    *,
    strategy: str = "source_val_macro_f1",
    grid_size: int = 181,
) -> tuple[float, float]:
    """Choose a threshold using LABELLED SOURCE VALIDATION data only."""
    y = np.asarray(y_true).astype(int)
    score = np.asarray(positive_score, dtype=float)
    if strategy == "fixed_0.5":
        pred = (score >= 0.5).astype(int)
        return 0.5, float(f1_score(y, pred, average="macro", zero_division=0))

    if strategy not in {"source_val_macro_f1", "source_val_balanced_accuracy"}:
        raise ValueError(
            "strategy must be fixed_0.5, source_val_macro_f1, or source_val_balanced_accuracy"
        )

    thresholds = np.linspace(0.05, 0.95, int(grid_size))
    best_t, best_value = 0.5, -np.inf
    for t in thresholds:
        pred = (score >= t).astype(int)
        if strategy == "source_val_macro_f1":
            value = f1_score(y, pred, average="macro", zero_division=0)
        else:
            value = balanced_accuracy_score(y, pred)
        if value > best_value + 1e-12 or (
            abs(value - best_value) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)
        ):
            best_t, best_value = float(t), float(value)
    return best_t, best_value
