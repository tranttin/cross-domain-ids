import numpy as np
import pandas as pd

from ids.preprocessing.cross_domain import prepare_legacy_notebook_protocol, prepare_clean_source_target


def make_df(n, year, seed):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "A": rng.normal(loc=0 if year == 2018 else 10, scale=1, size=n),
        "B": rng.normal(size=n),
        "C": rng.normal(size=n),
    })
    y = np.array([0, 1] * (n // 2))
    X["Label"] = y
    return X


def test_legacy_shapes_and_2018_reference_fit():
    df18 = make_df(100, 2018, 1)
    df17 = make_df(80, 2017, 2)
    p = prepare_legacy_notebook_protocol(df17, df18, feature_order="sorted", seed=42, threshold=0.0)
    assert p.X18_train.shape[0] == 80
    assert p.X18_test.shape[0] == 20
    assert p.X17_all.shape[0] == 80
    # Because scaling is fit on 2018 but 2017 A is shifted by ~10, transformed 2017
    # should remain far from the 2018 center. This guards the legacy reference-domain behavior.
    assert abs(np.median(p.X17_all[:, 0])) > 2


def test_clean_source_fit_changes_with_direction():
    df18 = make_df(100, 2018, 1)
    df17 = make_df(80, 2017, 2)
    p17 = prepare_clean_source_target(df17, df18, "2017", "2018", seed=42, threshold=0.0)
    # Clean source scaling centers the 2017 source rather than the 2018 target.
    assert abs(np.median(p17.X_source_train[:, 0])) < 1
