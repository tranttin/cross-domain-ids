from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler


@dataclass
class InDomainSplit:
    X_train_ml: np.ndarray
    X_test_ml: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_cnn: np.ndarray
    X_test_cnn: np.ndarray
    selector: VarianceThreshold
    scaler: RobustScaler
    n_classes: int
    input_feature_names: list[str]
    selected_feature_names: list[str]


def preprocess_legacy_2018(
    data: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    variance_threshold: float = 0.01,
    quantile_range=(5, 95),
) -> InDomainSplit:
    X = data.drop("Label", axis=1)
    y = data["Label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    selector = VarianceThreshold(threshold=variance_threshold)
    X_train_f = selector.fit_transform(X_train)
    X_test_f = selector.transform(X_test)
    selected = [c for c, keep in zip(X.columns, selector.get_support()) if keep]

    scaler = RobustScaler(quantile_range=quantile_range)
    X_train_s = scaler.fit_transform(X_train_f)
    X_test_s = scaler.transform(X_test_f)

    return InDomainSplit(
        X_train_ml=X_train_s,
        X_test_ml=X_test_s,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        X_train_cnn=X_train_s[..., np.newaxis],
        X_test_cnn=X_test_s[..., np.newaxis],
        selector=selector,
        scaler=scaler,
        n_classes=int(y.nunique()),
        input_feature_names=list(X.columns),
        selected_feature_names=selected,
    )


@dataclass
class CleanBinaryInDomainSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    selector: VarianceThreshold
    scaler: RobustScaler
    input_feature_names: list[str]
    selected_feature_names: list[str]


def preprocess_clean_binary_in_domain(
    data: pd.DataFrame,
    dataset_name: str,
    *,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
    variance_threshold: float = 0.01,
    quantile_range=(5, 95),
) -> CleanBinaryInDomainSplit:
    """Leakage-resistant binary in-domain split used by the journal benchmark.

    The selector and scaler are fitted on training data only. Validation and test
    partitions are stratified and never used to fit preprocessing parameters.
    Labels are collapsed to Benign=0 / Attack=1 through the same dataset adapter
    used by clean cross-domain experiments.
    """
    from ids.preprocessing.cross_domain import binary_labels_for_dataset

    if "Label" not in data.columns:
        raise KeyError("Prepared dataset must contain a 'Label' column")
    X = data.drop(columns=["Label"]).select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y_all = binary_labels_for_dataset(data.loc[X.index], dataset_name)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )
    val_fraction_of_dev = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_fraction_of_dev, random_state=seed, stratify=y_dev
    )

    selector = VarianceThreshold(threshold=variance_threshold)
    X_train_f = selector.fit_transform(X_train)
    X_val_f = selector.transform(X_val)
    X_test_f = selector.transform(X_test)
    selected = [c for c, keep in zip(X.columns, selector.get_support()) if keep]

    scaler = RobustScaler(quantile_range=quantile_range)
    X_train_s = scaler.fit_transform(X_train_f)
    X_val_s = scaler.transform(X_val_f)
    X_test_s = scaler.transform(X_test_f)

    return CleanBinaryInDomainSplit(
        X_train=X_train_s.astype(np.float32),
        X_val=X_val_s.astype(np.float32),
        X_test=X_test_s.astype(np.float32),
        y_train=np.asarray(y_train, dtype=np.int32),
        y_val=np.asarray(y_val, dtype=np.int32),
        y_test=np.asarray(y_test, dtype=np.int32),
        selector=selector,
        scaler=scaler,
        input_feature_names=list(X.columns),
        selected_feature_names=selected,
    )


def preprocess_clean_binary_in_domain_fixed_test(
    dev_data: pd.DataFrame,
    test_data: pd.DataFrame,
    dataset_name: str,
    *,
    validation_size: float = 0.1,
    seed: int = 42,
    variance_threshold: float = 0.01,
    quantile_range=(5, 95),
) -> CleanBinaryInDomainSplit:
    """In-domain protocol with externally frozen final test data."""
    from ids.preprocessing.cross_domain import binary_labels_for_dataset

    for frame in (dev_data, test_data):
        if "Label" not in frame.columns:
            raise KeyError("Prepared dataset must contain a 'Label' column")
    Xd = dev_data.drop(columns=["Label"]).select_dtypes(include=[np.number]).copy()
    Xt = test_data.drop(columns=["Label"]).select_dtypes(include=[np.number]).copy()
    common = [c for c in Xd.columns if c in Xt.columns]
    if not common:
        raise ValueError("No common numeric features between dev and fixed test data")
    Xd = Xd[common].replace([np.inf, -np.inf], np.nan)
    Xt = Xt[common].replace([np.inf, -np.inf], np.nan)
    valid_d = ~Xd.isna().any(axis=1)
    valid_t = ~Xt.isna().any(axis=1)
    Xd = Xd.loc[valid_d]
    Xt = Xt.loc[valid_t]
    yd = binary_labels_for_dataset(dev_data.loc[Xd.index], dataset_name)
    yt = binary_labels_for_dataset(test_data.loc[Xt.index], dataset_name)

    X_train, X_val, y_train, y_val = train_test_split(
        Xd, yd, test_size=validation_size, random_state=seed, stratify=yd
    )
    selector = VarianceThreshold(threshold=variance_threshold)
    X_train_f = selector.fit_transform(X_train)
    X_val_f = selector.transform(X_val)
    X_test_f = selector.transform(Xt)
    selected = [c for c, keep in zip(common, selector.get_support()) if keep]
    scaler = RobustScaler(quantile_range=quantile_range)
    X_train_s = scaler.fit_transform(X_train_f)
    X_val_s = scaler.transform(X_val_f)
    X_test_s = scaler.transform(X_test_f)
    return CleanBinaryInDomainSplit(
        X_train=X_train_s.astype(np.float32), X_val=X_val_s.astype(np.float32),
        X_test=X_test_s.astype(np.float32), y_train=np.asarray(y_train, dtype=np.int32),
        y_val=np.asarray(y_val, dtype=np.int32), y_test=np.asarray(yt, dtype=np.int32),
        selector=selector, scaler=scaler, input_feature_names=common,
        selected_feature_names=selected,
    )
