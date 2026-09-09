from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def balanced_class_weights(y: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency class weights from LABELLED SOURCE data only."""
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def class_weight_vector(y: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    """Expand a class-weight dictionary into per-example float32 weights."""
    y = np.asarray(y).astype(int)
    return np.asarray([weights[int(v)] for v in y], dtype=np.float32)
