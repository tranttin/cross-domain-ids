import numpy as np
from ids.evaluation.thresholds import choose_binary_threshold

def test_source_val_optimal_threshold_can_rescue_macro_f1_from_fixed_half():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.02, 0.04, 0.08, 0.18, 0.22, 0.30])
    t, macro = choose_binary_threshold(y, score, strategy="source_val_macro_f1")
    assert t < 0.5
    assert macro > 0.95
