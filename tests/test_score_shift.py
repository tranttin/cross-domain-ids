import numpy as np

from ids.evaluation.score_shift import (
    build_score_shift_diagnostic,
    choose_target_oracle_threshold,
    positive_class_score,
)


class ReversedClassProbaModel:
    classes_ = np.array([1, 0])

    def predict_proba(self, X):
        # First column is class 1 because classes_ is [1, 0].
        p1 = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([p1, 1.0 - p1])


def test_positive_class_score_resolves_class_index_instead_of_assuming_column_one():
    model = ReversedClassProbaModel()
    score, meta = positive_class_score(model, np.array([0.2, 0.8]))
    assert np.allclose(score, [0.2, 0.8])
    assert meta.positive_class_index == 0
    assert meta.classes == [1, 0]


def test_target_oracle_threshold_exposes_score_location_shift():
    y = np.array([0, 0, 0, 1, 1, 1])
    # Perfect ranking, but all scores are far below a source threshold such as 0.3.
    score = np.array([0.001, 0.002, 0.003, 0.020, 0.030, 0.040])
    oracle = choose_target_oracle_threshold(y, score)
    assert oracle["macro_f1"] > 0.99
    assert oracle["threshold"] < 0.3


def test_score_shift_diagnostic_flags_threshold_collapse_and_orientation_inversion():
    sy = np.array([0, 0, 1, 1])
    sv = np.array([0.1, 0.2, 0.8, 0.9])
    ty = np.array([0, 0, 1, 1])

    shifted = np.array([0.001, 0.002, 0.02, 0.03])
    d = build_score_shift_diagnostic(
        source_val_score=sv,
        source_val_labels=sy,
        source_threshold=0.5,
        target_score=shifted,
        target_labels=ty,
    )
    assert d["unlabelled_threshold_collapse_suspect"] is True
    assert d["diagnostic_target_auroc_raw"] == 1.0
    assert d["diagnostic_target_oracle_macro_f1"] > 0.99

    inverted = np.array([0.9, 0.8, 0.2, 0.1])
    d2 = build_score_shift_diagnostic(
        source_val_score=sv,
        source_val_labels=sy,
        source_threshold=0.5,
        target_score=inverted,
        target_labels=ty,
    )
    assert d2["diagnostic_target_auroc_raw"] == 0.0
    assert d2["diagnostic_target_auroc_reversed"] == 1.0
    assert d2["diagnostic_orientation_suspect"] is True
