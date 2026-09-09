from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ids.preprocessing.cross_domain import binary_labels_for_dataset


@dataclass
class DiagnosticTables:
    feature_stats: pd.DataFrame
    class_distribution: pd.DataFrame
    out_of_range: pd.DataFrame
    missing_or_invalid: pd.DataFrame


def _safe_numeric(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in features:
        out[c] = pd.to_numeric(df[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def _stats(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for c in frame.columns:
        s = frame[c]
        finite = s.dropna().to_numpy(dtype=float)
        row = {
            "feature": c,
            f"{prefix}_n": int(len(s)),
            f"{prefix}_missing": int(s.isna().sum()),
            f"{prefix}_missing_rate": float(s.isna().mean()),
            f"{prefix}_unique": int(s.nunique(dropna=True)),
        }
        if len(finite):
            row.update({
                f"{prefix}_min": float(np.min(finite)),
                f"{prefix}_p01": float(np.quantile(finite, 0.01)),
                f"{prefix}_median": float(np.median(finite)),
                f"{prefix}_p99": float(np.quantile(finite, 0.99)),
                f"{prefix}_max": float(np.max(finite)),
                f"{prefix}_mean": float(np.mean(finite)),
                f"{prefix}_std": float(np.std(finite)),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame, dataset_name: str, role: str) -> pd.DataFrame:
    y = binary_labels_for_dataset(df, dataset_name)
    n = len(y)
    return pd.DataFrame([
        {"dataset": dataset_name, "role": role, "label": 0, "name": "benign", "count": int((y == 0).sum()), "fraction": float((y == 0).mean()) if n else np.nan},
        {"dataset": dataset_name, "role": role, "label": 1, "name": "attack", "count": int((y == 1).sum()), "fraction": float((y == 1).mean()) if n else np.nan},
    ])


def diagnose_aligned_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str,
    target_name: str,
    *,
    seed: int = 42,
    source_train_fraction: float = 0.7,
) -> DiagnosticTables:
    features = [c for c in source_df.columns if c != "Label"]
    if features != [c for c in target_df.columns if c != "Label"]:
        raise ValueError("Diagnostics require frames aligned to the same ordered features")

    ys = binary_labels_for_dataset(source_df, source_name)
    idx = np.arange(len(source_df))
    train_idx, _ = train_test_split(
        idx, train_size=source_train_fraction, random_state=seed, stratify=ys
    )
    source_train = _safe_numeric(source_df.iloc[train_idx], features)
    target_all = _safe_numeric(target_df, features)

    ss = _stats(source_train, "source")
    ts = _stats(target_all, "target")
    stats = ss.merge(ts, on="feature", how="outer")
    eps = 1e-12
    stats["median_shift_vs_source_iqr_proxy"] = (
        (stats["target_median"] - stats["source_median"]).abs()
        / ((stats["source_p99"] - stats["source_p01"]).abs() + eps)
    )
    stats["target_median_over_source_p99_abs"] = (
        stats["target_median"].abs() / (stats["source_p99"].abs() + eps)
    )

    oor_rows = []
    invalid_rows = []
    for c in features:
        s = source_train[c].dropna().to_numpy(float)
        t = target_all[c]
        tf = t.dropna().to_numpy(float)
        invalid_rows.append({
            "feature": c,
            "source_missing_rate": float(source_train[c].isna().mean()),
            "target_missing_rate": float(t.isna().mean()),
            "source_constant": bool(source_train[c].nunique(dropna=True) <= 1),
            "target_constant": bool(t.nunique(dropna=True) <= 1),
        })
        if len(s) == 0 or len(tf) == 0:
            oor_rows.append({"feature": c, "target_outside_source_minmax": np.nan, "target_outside_source_p01_p99": np.nan})
            continue
        mn, mx = float(np.min(s)), float(np.max(s))
        p01, p99 = np.quantile(s, [0.01, 0.99])
        oor_rows.append({
            "feature": c,
            "target_outside_source_minmax": float(np.mean((tf < mn) | (tf > mx))),
            "target_outside_source_p01_p99": float(np.mean((tf < p01) | (tf > p99))),
        })

    classes = pd.concat([
        class_distribution(source_df, source_name, "source_all"),
        class_distribution(target_df, target_name, "target_all_evaluation_only"),
    ], ignore_index=True)
    return DiagnosticTables(
        feature_stats=stats,
        class_distribution=classes,
        out_of_range=pd.DataFrame(oor_rows),
        missing_or_invalid=pd.DataFrame(invalid_rows),
    )


def scaled_shift_table(split) -> pd.DataFrame:
    """Compare source-train and held-out target after source-fitted selector/scaler."""
    rows = []
    Xs = np.asarray(split.X_source_train)
    Xt = np.asarray(split.X_target_test)
    for j, name in enumerate(split.selected_feature_names):
        s = Xs[:, j]
        t = Xt[:, j]
        p01, p99 = np.quantile(s, [0.01, 0.99])
        mn, mx = np.min(s), np.max(s)
        rows.append({
            "feature": name,
            "source_scaled_min": float(mn),
            "source_scaled_p01": float(p01),
            "source_scaled_median": float(np.median(s)),
            "source_scaled_p99": float(p99),
            "source_scaled_max": float(mx),
            "target_scaled_min": float(np.min(t)),
            "target_scaled_p01": float(np.quantile(t, 0.01)),
            "target_scaled_median": float(np.median(t)),
            "target_scaled_p99": float(np.quantile(t, 0.99)),
            "target_scaled_max": float(np.max(t)),
            "target_outside_scaled_source_minmax": float(np.mean((t < mn) | (t > mx))),
            "target_outside_scaled_source_p01_p99": float(np.mean((t < p01) | (t > p99))),
        })
    return pd.DataFrame(rows)
