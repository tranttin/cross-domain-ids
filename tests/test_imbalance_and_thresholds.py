import numpy as np

from ids.training.imbalance import balanced_class_weights, class_weight_vector
from ids.evaluation.thresholds import choose_binary_threshold


def test_balanced_weights_source_only_shape_and_direction():
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    w = balanced_class_weights(y)
    assert w[0] > w[1]
    sw = class_weight_vector(y, w)
    assert sw.shape == y.shape
    # Balanced weighting gives the two classes equal aggregate contribution.
    assert np.isclose(sw[y == 0].sum(), sw[y == 1].sum())


def test_threshold_selection_can_move_from_half():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.10, 0.20, 0.30, 0.35, 0.40, 0.45])
    t, value = choose_binary_threshold(y, score, strategy="source_val_macro_f1")
    assert t < 0.5
    assert value > 0.9


def test_fixed_threshold_is_exactly_half():
    y = np.array([0, 1])
    score = np.array([0.2, 0.8])
    t, _ = choose_binary_threshold(y, score, strategy="fixed_0.5")
    assert t == 0.5
