from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SupervisedArrays:
    """Minimal model-facing contract for supervised IDS experiments."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


@dataclass
class UDAArrays:
    """Model-facing contract for unsupervised domain adaptation.

    Target class labels are deliberately absent from the training side of the
    contract. `y_target_test` exists only for final evaluation.
    """
    X_source_train: np.ndarray
    y_source_train: np.ndarray
    X_source_val: np.ndarray
    y_source_val: np.ndarray
    X_target_unlabeled: np.ndarray
    X_target_test: np.ndarray
    y_target_test: np.ndarray
    feature_names: list[str]
