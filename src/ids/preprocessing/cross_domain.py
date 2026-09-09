from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler

from ids.data.feature_schema import feature_overlap


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().replace(" ", "_") for c in out.columns]
    return out


def _binary_labels_2017(df: pd.DataFrame) -> np.ndarray:
    return (df["Label"].astype(int) != 0).astype(int).to_numpy()


def _binary_labels_2018(df: pd.DataFrame) -> np.ndarray:
    lbl = df["Label"]
    if lbl.dtype == "O":
        s = lbl.astype(str).str.upper()
        if any("BENIGN" in u for u in s.unique()):
            return (~s.str.contains("BENIGN")).astype(int).to_numpy()
        return (s.astype(int) != 0).astype(int).to_numpy()
    return (lbl.astype(int) != 0).astype(int).to_numpy()


def binary_labels_for_dataset(df: pd.DataFrame, name: str) -> np.ndarray:
    """Public binary-label adapter used by clean diagnostics/runners.

    Prepared third-domain datasets must follow the research contract: Label=0 for
    benign and any non-zero value for attack. CIC-IDS2017/2018 retain their recovered
    notebook-specific handling.
    """
    if "2017" in str(name):
        return _binary_labels_2017(df)
    if "2018" in str(name):
        return _binary_labels_2018(df)
    return (df["Label"].astype(int) != 0).astype(int).to_numpy()


def common_feature_names(df17: pd.DataFrame, df18: pd.DataFrame, order: str = "sorted") -> list[str]:
    cols17 = set(df17.columns)
    cols18 = set(df18.columns)
    common = (cols17 & cols18) - {"Label", "Label_bin"}
    if order == "legacy_set":
        # Exact notebook expression. Python hash randomization makes this unstable
        # across processes; every run therefore saves the resulting order.
        return list(common)
    if order == "sorted":
        return sorted(common)
    if order.startswith("file:"):
        path = order.split(":", 1)[1]
        names = [x.strip() for x in open(path, encoding="utf-8") if x.strip()]
        missing = [x for x in names if x not in common]
        if missing:
            raise ValueError(f"Feature-order file contains non-common columns: {missing[:5]}")
        return names
    raise ValueError("order must be 'legacy_set', 'sorted', or 'file:<path>'")


@dataclass
class LegacyNotebookCrossDomain:
    feature_names: list[str]
    selected_feature_names: list[str]
    X18_train: np.ndarray
    X18_test: np.ndarray
    y18_train: np.ndarray
    y18_test: np.ndarray
    X18_all: np.ndarray
    y18_all: np.ndarray
    X17_train: np.ndarray
    X17_test: np.ndarray
    y17_train: np.ndarray
    y17_test: np.ndarray
    X17_all: np.ndarray
    y17_all: np.ndarray
    selector: VarianceThreshold
    scaler: RobustScaler


def prepare_legacy_notebook_protocol(
    df17: pd.DataFrame,
    df18: pd.DataFrame,
    feature_order: str = "legacy_set",
    test_size: float = 0.2,
    seed: int = 42,
    threshold: float = 0.01,
    quantile_range=(5, 95),
) -> LegacyNotebookCrossDomain:
    """Reproduce the cross-domain preprocessing cells in `cross-domain.ipynb`.

    Critical legacy behavior preserved intentionally:
    - common features come from a Python set intersection;
    - both 2017 and 2018 are split, but selector/scaler are ALWAYS fit on 2018 train;
    - 2017 full is also transformed for the 2018->2017 evaluation/target pool;
    - 2018 full is transformed for the 2017-source DANN target pool.

    This function exists for historical reproduction only.
    """
    df17 = _normalize_columns(df17)
    df18 = _normalize_columns(df18)
    features = common_feature_names(df17, df18, order=feature_order)

    X17 = df17[features].to_numpy(dtype=np.float32)
    X18 = df18[features].to_numpy(dtype=np.float32)
    y17 = _binary_labels_2017(df17)
    y18 = _binary_labels_2018(df18)

    X18_tr, X18_te, y18_tr, y18_te = train_test_split(
        X18, y18, test_size=test_size, stratify=y18, random_state=seed
    )
    X17_tr, X17_te, y17_tr, y17_te = train_test_split(
        X17, y17, test_size=test_size, stratify=y17, random_state=seed
    )

    selector = VarianceThreshold(threshold=threshold).fit(X18_tr)
    mask = selector.get_support()
    selected = [f for f, keep in zip(features, mask) if keep]

    X18_tr_f = selector.transform(X18_tr)
    X18_te_f = selector.transform(X18_te)
    X18_all_f = selector.transform(X18)
    X17_tr_f = selector.transform(X17_tr)
    X17_te_f = selector.transform(X17_te)
    X17_all_f = selector.transform(X17)

    scaler = RobustScaler(quantile_range=quantile_range).fit(X18_tr_f)
    transform = scaler.transform
    return LegacyNotebookCrossDomain(
        feature_names=features,
        selected_feature_names=selected,
        X18_train=transform(X18_tr_f),
        X18_test=transform(X18_te_f),
        y18_train=y18_tr,
        y18_test=y18_te,
        X18_all=transform(X18_all_f),
        y18_all=y18,
        X17_train=transform(X17_tr_f),
        X17_test=transform(X17_te_f),
        y17_train=y17_tr,
        y17_test=y17_te,
        X17_all=transform(X17_all_f),
        y17_all=y17,
        selector=selector,
        scaler=scaler,
    )


@dataclass
class CleanSourceTargetSplit:
    feature_names: list[str]
    selected_feature_names: list[str]
    X_source_train: np.ndarray
    y_source_train: np.ndarray
    X_source_val: np.ndarray
    y_source_val: np.ndarray
    X_source_test: np.ndarray
    y_source_test: np.ndarray
    X_target_unlabeled: np.ndarray
    X_target_test: np.ndarray
    y_target_test: np.ndarray
    selector: VarianceThreshold
    scaler: RobustScaler


def prepare_clean_source_target(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str,
    target_name: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
    threshold: float = 0.01,
    quantile_range=(5, 95),
    feature_alignment: str = "exact",
) -> CleanSourceTargetSplit:
    """Leakage-resistant UDA split.

    Source labels are available for train/validation/test. Target is split into an
    unlabeled adaptation pool and a held-out labelled evaluation set. Feature
    selection/scaling are fit on SOURCE TRAIN only. Target labels are returned only
    for final evaluation and must not be used by the training code.
    """
    source_df = _normalize_columns(source_df)
    target_df = _normalize_columns(target_df)
    if feature_alignment == "exact":
        features = common_feature_names(source_df, target_df, order="sorted")
        source_columns = features
        target_columns = features
    elif feature_alignment == "canonical":
        ov = feature_overlap(source_df.columns, target_df.columns)
        features = ov.common
        source_columns = [ov.source_map[k] for k in features]
        target_columns = [ov.target_map[k] for k in features]
    else:
        raise ValueError("feature_alignment must be 'exact' or 'canonical'")
    if not features:
        raise ValueError("No common features between source and target")

    Xs = source_df[source_columns].to_numpy(dtype=np.float32)
    Xt = target_df[target_columns].to_numpy(dtype=np.float32)
    ys = binary_labels_for_dataset(source_df, source_name)
    yt = binary_labels_for_dataset(target_df, target_name)

    Xs_dev, Xs_test, ys_dev, ys_test = train_test_split(
        Xs, ys, test_size=test_size, stratify=ys, random_state=seed
    )
    val_fraction_of_dev = validation_size / (1.0 - test_size)
    Xs_train, Xs_val, ys_train, ys_val = train_test_split(
        Xs_dev, ys_dev, test_size=val_fraction_of_dev, stratify=ys_dev, random_state=seed
    )
    Xt_adapt, Xt_test, _, yt_test = train_test_split(
        Xt, yt, test_size=test_size, stratify=yt, random_state=seed
    )

    selector = VarianceThreshold(threshold=threshold).fit(Xs_train)
    mask = selector.get_support()
    selected = [f for f, keep in zip(features, mask) if keep]
    scaler = RobustScaler(quantile_range=quantile_range).fit(selector.transform(Xs_train))

    def tx(X):
        return scaler.transform(selector.transform(X))

    return CleanSourceTargetSplit(
        feature_names=features,
        selected_feature_names=selected,
        X_source_train=tx(Xs_train), y_source_train=ys_train,
        X_source_val=tx(Xs_val), y_source_val=ys_val,
        X_source_test=tx(Xs_test), y_source_test=ys_test,
        X_target_unlabeled=tx(Xt_adapt),
        X_target_test=tx(Xt_test), y_target_test=yt_test,
        selector=selector, scaler=scaler,
    )


# Backward-compatible aliases kept for any local scripts created from v0.1.
@dataclass
class CrossDomainData:
    feature_names: list[str]
    X17: np.ndarray
    y17: np.ndarray
    X18: np.ndarray
    y18: np.ndarray


def prepare_common_features(df17: pd.DataFrame, df18: pd.DataFrame, feature_order: str = "sorted") -> CrossDomainData:
    df17 = _normalize_columns(df17); df18 = _normalize_columns(df18)
    features = common_feature_names(df17, df18, feature_order)
    return CrossDomainData(
        features,
        df17[features].to_numpy(dtype=np.float32), _binary_labels_2017(df17),
        df18[features].to_numpy(dtype=np.float32), _binary_labels_2018(df18),
    )


@dataclass
class SourceFitTransform:
    Xs_train: np.ndarray
    Xs_test: np.ndarray
    ys_train: np.ndarray
    ys_test: np.ndarray
    Xt_all: np.ndarray
    selector: VarianceThreshold
    scaler: RobustScaler


def fit_on_source(Xs, ys, Xt, test_size=0.2, seed=42, threshold=0.01, quantile_range=(5, 95)):
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=test_size, stratify=ys, random_state=seed)
    selector = VarianceThreshold(threshold=threshold).fit(Xtr)
    Xtr_f, Xte_f, Xt_f = selector.transform(Xtr), selector.transform(Xte), selector.transform(Xt)
    scaler = RobustScaler(quantile_range=quantile_range).fit(Xtr_f)
    return SourceFitTransform(scaler.transform(Xtr_f), scaler.transform(Xte_f), ytr, yte, scaler.transform(Xt_f), selector, scaler)


def prepare_clean_source_target_fixed_test(
    source_df: pd.DataFrame,
    target_adapt_df: pd.DataFrame,
    target_test_df: pd.DataFrame,
    source_name: str,
    target_name: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
    threshold: float = 0.01,
    quantile_range=(5, 95),
    feature_alignment: str = "exact",
) -> CleanSourceTargetSplit:
    """Clean UDA split with an externally frozen target test set.

    `target_adapt_df` is treated as fully unlabeled during training. `target_test_df`
    is transformed with preprocessing fitted on SOURCE TRAIN and is touched only at
    final evaluation time. This helper is the preferred final-paper protocol after
    hyperparameter choices have been frozen.
    """
    source_df = _normalize_columns(source_df)
    target_adapt_df = _normalize_columns(target_adapt_df)
    target_test_df = _normalize_columns(target_test_df)

    if feature_alignment == "exact":
        common_adapt = set(common_feature_names(source_df, target_adapt_df, order="sorted"))
        common_test = set(common_feature_names(source_df, target_test_df, order="sorted"))
        features = sorted(common_adapt & common_test)
        source_columns = features
        adapt_columns = features
        test_columns = features
    elif feature_alignment == "canonical":
        ov_a = feature_overlap(source_df.columns, target_adapt_df.columns)
        ov_t = feature_overlap(source_df.columns, target_test_df.columns)
        features = sorted(set(ov_a.common) & set(ov_t.common))
        source_columns = [ov_a.source_map[k] for k in features]
        adapt_columns = [ov_a.target_map[k] for k in features]
        test_columns = [ov_t.target_map[k] for k in features]
    else:
        raise ValueError("feature_alignment must be 'exact' or 'canonical'")
    if not features:
        raise ValueError("No common features across source, target adaptation, and target test")

    Xs = source_df[source_columns].to_numpy(dtype=np.float32)
    Xa = target_adapt_df[adapt_columns].to_numpy(dtype=np.float32)
    Xt = target_test_df[test_columns].to_numpy(dtype=np.float32)
    ys = binary_labels_for_dataset(source_df, source_name)
    yt = binary_labels_for_dataset(target_test_df, target_name)

    Xs_dev, Xs_test, ys_dev, ys_test = train_test_split(
        Xs, ys, test_size=test_size, stratify=ys, random_state=seed
    )
    val_fraction_of_dev = validation_size / (1.0 - test_size)
    Xs_train, Xs_val, ys_train, ys_val = train_test_split(
        Xs_dev, ys_dev, test_size=val_fraction_of_dev,
        stratify=ys_dev, random_state=seed,
    )

    selector = VarianceThreshold(threshold=threshold).fit(Xs_train)
    mask = selector.get_support()
    selected = [f for f, keep in zip(features, mask) if keep]
    scaler = RobustScaler(quantile_range=quantile_range).fit(selector.transform(Xs_train))

    def tx(X):
        return scaler.transform(selector.transform(X))

    return CleanSourceTargetSplit(
        feature_names=features,
        selected_feature_names=selected,
        X_source_train=tx(Xs_train), y_source_train=ys_train,
        X_source_val=tx(Xs_val), y_source_val=ys_val,
        X_source_test=tx(Xs_test), y_source_test=ys_test,
        X_target_unlabeled=tx(Xa),
        X_target_test=tx(Xt), y_target_test=yt,
        selector=selector, scaler=scaler,
    )
