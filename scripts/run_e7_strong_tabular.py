#!/usr/bin/env python3
"""E7 strong tabular baselines for the IDS journal protocol.

Purpose
-------
Add XGBoost and CatBoost without changing E1--E4 deep-model training.
The script follows the paper's frozen source-anchored protocol:
  * binary task: benign=0, attack=1;
  * source train/validation/test split = 70/10/20 for a single CSV;
  * target adaptation/test split = 80/20 for the related CIC transfer;
  * source-validation threshold selection only;
  * optional label-free GMM-logit threshold for enterprise->IoT transfer;
  * no target labels are used for fitting models, thresholds, or hyperparameters.

The script is deliberately standalone because the current manuscript package does
not contain the live IDS-2026-Research Python source tree. It can be copied to
that repository as scripts/run_e7_strong_tabular.py. If the repository already
has canonical-feature utilities, prefer those utilities over the generic YAML
mapping fallback implemented here.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    XGBClassifier = None
    XGB_IMPORT_ERROR = exc
else:
    XGB_IMPORT_ERROR = None

try:
    from catboost import CatBoostClassifier
except Exception as exc:  # pragma: no cover
    CatBoostClassifier = None
    CAT_IMPORT_ERROR = exc
else:
    CAT_IMPORT_ERROR = None


CANONICAL_IOT_FEATURES = [
    "backward_packets_per_second",
    "flow_packets_per_second",
    "forward_packets_per_second",
    "packet_length_max",
    "packet_length_mean",
    "packet_length_min",
    "packet_length_std",
]

DEFAULT_PATHS = {
    "2017": Path("data/processed/cicids2017_balanced_77feats.csv"),
    "2018": Path("data/processed/cleaned_data_sampled.csv"),
    "iot_dev": Path("data/processed/ciciot2023_dev.csv"),
    "iot_test": Path("data/processed/ciciot2023_final_test.csv"),
    "mapping": Path("configs/features/canonical_features.yaml"),
}

SCENARIOS = {
    "E1_2017": ("in_domain", "2017", "2017"),
    "E1_2018": ("in_domain", "2018", "2018"),
    "E1_IoT": ("iot_in_domain", "CICIoT2023", "CICIoT2023"),
    "E2_17to18": ("related_transfer", "2017", "2018"),
    "E2_18to17": ("related_transfer", "2018", "2017"),
    "E3_17toIoT": ("iot_transfer", "2017", "CICIoT2023"),
    "E3_18toIoT": ("iot_transfer", "2018", "CICIoT2023"),
}

LABEL_CANDIDATES = ["label_bin", "label", "class", "target", "attack"]
DROP_NAME_HINTS = {
    "flow_id", "flowid", "timestamp", "src_ip", "dst_ip", "source_ip",
    "destination_ip", "src_port", "dst_port", "source_port", "destination_port",
}


def norm_name(x: str) -> str:
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


def find_label_column(df: pd.DataFrame) -> str:
    normalized = {norm_name(c): c for c in df.columns}
    for cand in LABEL_CANDIDATES:
        if cand in normalized:
            return normalized[cand]
    for c in df.columns:
        n = norm_name(c)
        if "label" in n or n in {"y", "is_attack", "attack_label"}:
            return c
    raise ValueError(f"Could not identify label column. Columns include: {list(df.columns[:20])}")


def to_binary_labels(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce").fillna(0).to_numpy()
        unique = set(np.unique(v).tolist())
        if unique.issubset({0, 1, 0.0, 1.0}):
            return v.astype(np.int8)
        # Frozen CIC prepared files use 0 as benign and positive integers as attacks.
        return (v != 0).astype(np.int8)
    t = s.astype(str).str.strip().str.upper()
    benign = t.str.contains(r"BENIGN|NORMAL", regex=True, na=False)
    return (~benign).astype(np.int8).to_numpy()


def load_csv(path: Path) -> Tuple[pd.DataFrame, np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, low_memory=False)
    label_col = find_label_column(df)
    y = to_binary_labels(df[label_col])
    return df, y, label_col


def numeric_feature_map(df: pd.DataFrame, label_col: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in df.columns:
        if c == label_col:
            continue
        n = norm_name(c)
        if n in DROP_NAME_HINTS or any(h in n for h in ["timestamp", "flow_id", "src_ip", "dst_ip"]):
            continue
        # Accept columns that are already numeric OR can mostly be coerced.
        if pd.api.types.is_numeric_dtype(df[c]):
            out[n] = c
        else:
            probe = pd.to_numeric(df[c].head(5000), errors="coerce")
            if float(probe.notna().mean()) >= 0.98:
                out[n] = c
    return out


def clean_numeric(df: pd.DataFrame, columns: Sequence[str], medians: Optional[pd.Series] = None):
    x = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median(axis=0).fillna(0.0)
    x = x.fillna(medians).astype(np.float32)
    return x.to_numpy(), medians


def split_source(df: pd.DataFrame, y: np.ndarray, seed: int):
    idx = np.arange(len(df))
    train_val, test = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y)
    y_tv = y[train_val]
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=seed, stratify=y_tv
    )  # 0.125*0.8 = 0.10 total
    return train, val, test


def split_target(df: pd.DataFrame, y: np.ndarray, seed: int):
    idx = np.arange(len(df))
    adapt, test = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y)
    return adapt, test


def find_subtree_for_canonical(obj, canonical: str):
    target = norm_name(canonical)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if norm_name(k) == target:
                return v
        for _, v in obj.items():
            found = find_subtree_for_canonical(v, canonical)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                for key in ("canonical", "canonical_name", "name", "feature"):
                    if key in item and norm_name(item[key]) == target:
                        return item
                found = find_subtree_for_canonical(item, canonical)
                if found is not None:
                    return found
    return None


def flatten_scalars(obj, prefix=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten_scalars(v, prefix + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten_scalars(v, prefix + (str(i),))
    elif isinstance(obj, (str, int, float)):
        yield prefix, str(obj)


def domain_aliases(domain: str) -> List[str]:
    d = domain.lower()
    if "2017" in d:
        return ["2017", "cicids2017", "cic_ids2017"]
    if "2018" in d:
        return ["2018", "cse_cic_ids2018", "cicids2018", "cse-cic-ids2018"]
    return ["ciciot2023", "iot2023", "iot_2023", "iot"]


def resolve_canonical_columns(
    df: pd.DataFrame,
    domain: str,
    mapping_yaml: Path,
    canonicals: Sequence[str] = CANONICAL_IOT_FEATURES,
) -> Tuple[List[str], List[str]]:
    """Return actual dataframe columns and canonical labels in fixed order.

    The generic YAML parser supports common mapping layouts. If it cannot resolve
    a field, it also checks whether the processed CSV already exposes the canonical
    name directly. A hard failure is preferable to silently using a wrong feature.
    """
    colmap = {norm_name(c): c for c in df.columns}
    data = yaml.safe_load(mapping_yaml.read_text(encoding="utf-8")) if mapping_yaml.exists() else {}
    actual: List[str] = []
    used: List[str] = []
    aliases = domain_aliases(domain)

    for canonical in canonicals:
        cn = norm_name(canonical)
        if cn in colmap:
            actual.append(colmap[cn])
            used.append(canonical)
            continue
        subtree = find_subtree_for_canonical(data, canonical)
        candidates: List[str] = []
        if subtree is not None:
            for path, value in flatten_scalars(subtree):
                p = "_".join(norm_name(x) for x in path)
                if any(a in p for a in aliases):
                    candidates.append(value)
        resolved = None
        for value in candidates:
            vn = norm_name(value)
            if vn in colmap:
                resolved = colmap[vn]
                break
        if resolved is None:
            raise KeyError(
                f"Cannot resolve canonical feature '{canonical}' for domain {domain}. "
                f"Use the repository's canonical mapping utility or expose canonical columns in the processed CSV."
            )
        actual.append(resolved)
        used.append(canonical)
    return actual, used


def build_related_matrices(source_df, source_label, target_df, target_label, source_idx, target_idx):
    sm = numeric_feature_map(source_df, source_label)
    tm = numeric_feature_map(target_df, target_label)
    common = sorted(set(sm).intersection(tm))
    if not common:
        raise RuntimeError("No exact common numeric features found after column normalization")
    scol = [sm[n] for n in common]
    tcol = [tm[n] for n in common]

    x_src_raw, med = clean_numeric(source_df.iloc[source_idx], scol)
    # Medians need corresponding target names; use normalized-feature-indexed series.
    med_by_norm = {n: float(v) for n, v in zip(common, np.median(x_src_raw, axis=0))}
    tx = target_df.iloc[target_idx].loc[:, tcol].apply(pd.to_numeric, errors="coerce")
    tx = tx.replace([np.inf, -np.inf], np.nan)
    for n, c in zip(common, tcol):
        tx[c] = tx[c].fillna(med_by_norm[n])
    return x_src_raw.astype(np.float32), tx.to_numpy(dtype=np.float32), common


def fit_variance_selector(x_train: np.ndarray, *others: np.ndarray, threshold: float = 0.01):
    selector = VarianceThreshold(threshold=threshold)
    x_train2 = selector.fit_transform(x_train)
    transformed = [selector.transform(x) for x in others]
    return x_train2, transformed, selector


def best_macro_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    grid = np.linspace(0.005, 0.995, 199)
    best_t, best_v = 0.5, -1.0
    for t in grid:
        pred = (scores >= t).astype(np.int8)
        v = f1_score(y_true, pred, average="macro", zero_division=0)
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def gmm_logit_threshold(scores: np.ndarray, seed: int) -> Tuple[float, str]:
    eps = 1e-6
    p = np.clip(scores.astype(float), eps, 1 - eps)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, covariance_type="full", random_state=seed, n_init=5)
    gm.fit(z)
    means = gm.means_.ravel()
    vars_ = gm.covariances_.reshape(-1)
    weights = gm.weights_.ravel()
    order = np.argsort(means)
    m0, m1 = means[order]
    v0, v1 = max(vars_[order[0]], 1e-8), max(vars_[order[1]], 1e-8)
    w0, w1 = max(weights[order[0]], 1e-12), max(weights[order[1]], 1e-12)

    # Solve equality of weighted Gaussian densities in logit space.
    a = 1/(2*v1) - 1/(2*v0)
    b = m0/v0 - m1/v1
    c = (m1*m1)/(2*v1) - (m0*m0)/(2*v0) + math.log((w0*math.sqrt(v1))/(w1*math.sqrt(v0)))
    roots: List[float] = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            roots = [-c/b]
    else:
        disc = b*b - 4*a*c
        if disc >= 0:
            roots = [(-b + math.sqrt(disc))/(2*a), (-b - math.sqrt(disc))/(2*a)]
    between = [r for r in roots if m0 <= r <= m1]
    if between:
        zt = between[0]
        status = "ok"
    else:
        zt = float((m0 + m1) / 2.0)
        status = "midpoint_fallback"
    return float(sigmoid(zt)), status


def metrics_from_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "positive_prevalence": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(pred)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fpr": float(fp / max(tn + fp, 1)),
        "fnr": float(fn / max(tp + fn, 1)),
    }
    try:
        out["auprc"] = average_precision_score(y_true, scores)
    except Exception:
        out["auprc"] = float("nan")
    try:
        out["auroc"] = roc_auc_score(y_true, scores)
    except Exception:
        out["auroc"] = float("nan")
    return out


def make_models(seed: int):
    missing = []
    if XGBClassifier is None:
        missing.append(f"xgboost ({XGB_IMPORT_ERROR})")
    if CatBoostClassifier is None:
        missing.append(f"catboost ({CAT_IMPORT_ERROR})")
    if missing:
        raise RuntimeError("Missing E7 dependencies: " + "; ".join(missing))

    return {
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


@dataclass
class ScenarioData:
    scenario: str
    train_domain: str
    test_domain: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_target_adapt: Optional[np.ndarray]
    feature_names: List[str]
    threshold_strategy: str


def prepare_scenario(name: str, seed: int, paths: Dict[str, Path]) -> ScenarioData:
    mode, src_name, tgt_name = SCENARIOS[name]

    if mode == "in_domain":
        df, y, label = load_csv(paths[src_name])
        tr, va, te = split_source(df, y, seed)
        fmap = numeric_feature_map(df, label)
        names = sorted(fmap)
        cols = [fmap[n] for n in names]
        xtr, med = clean_numeric(df.iloc[tr], cols)
        xva, _ = clean_numeric(df.iloc[va], cols, med)
        xte, _ = clean_numeric(df.iloc[te], cols, med)
        xtr, [xva, xte], selector = fit_variance_selector(xtr, xva, xte)
        names = [n for n, keep in zip(names, selector.get_support()) if keep]
        return ScenarioData(name, src_name, tgt_name, xtr, y[tr], xva, y[va], xte, y[te], None, names, "source_val_macro_f1")

    if mode == "iot_in_domain":
        dev, ydev, ldev = load_csv(paths["iot_dev"])
        test, ytest, ltest = load_csv(paths["iot_test"])
        tr, va, _ = split_source(dev, ydev, seed)
        dm = numeric_feature_map(dev, ldev)
        tm = numeric_feature_map(test, ltest)
        names = sorted(set(dm).intersection(tm))
        if not names:
            raise RuntimeError("No common numeric IoT features found between DEV and FINAL_TEST")
        cols_dev = [dm[n] for n in names]
        cols_test = [tm[n] for n in names]
        xtr, med = clean_numeric(dev.iloc[tr], cols_dev)
        xva, _ = clean_numeric(dev.iloc[va], cols_dev, med)
        med_by_norm = {n: float(med.iloc[j]) for j, n in enumerate(names)}
        xte = test.loc[:, cols_test].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        for n, c in zip(names, cols_test):
            xte[c] = xte[c].fillna(med_by_norm[n])
        xtr, [xva, xte_np], selector = fit_variance_selector(xtr, xva, xte.to_numpy(np.float32))
        names = [n for n, keep in zip(names, selector.get_support()) if keep]
        return ScenarioData(name, src_name, tgt_name, xtr, ydev[tr], xva, ydev[va], xte_np, ytest, None, names, "source_val_macro_f1")

    if mode == "related_transfer":
        sdf, sy, slabel = load_csv(paths[src_name])
        tdf, ty, tlabel = load_csv(paths[tgt_name])
        tr, va, _ = split_source(sdf, sy, seed)
        ta, te = split_target(tdf, ty, seed)

        sm = numeric_feature_map(sdf, slabel)
        tm = numeric_feature_map(tdf, tlabel)
        names = sorted(set(sm).intersection(tm))
        if not names:
            raise RuntimeError("No related-domain common features")
        scols = [sm[n] for n in names]
        tcols = [tm[n] for n in names]
        xtr, med = clean_numeric(sdf.iloc[tr], scols)
        xva, _ = clean_numeric(sdf.iloc[va], scols, med)
        med_by_norm = {n: float(med.iloc[j]) for j, n in enumerate(names)}
        def target_matrix(index):
            x = tdf.iloc[index].loc[:, tcols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            for n, c in zip(names, tcols):
                x[c] = x[c].fillna(med_by_norm[n])
            return x.to_numpy(np.float32)
        xadapt = target_matrix(ta)
        xte = target_matrix(te)
        xtr, [xva, xadapt, xte], selector = fit_variance_selector(xtr, xva, xadapt, xte)
        names = [n for n, keep in zip(names, selector.get_support()) if keep]
        return ScenarioData(name, src_name, tgt_name, xtr, sy[tr], xva, sy[va], xte, ty[te], xadapt, names, "source_val_macro_f1")

    if mode == "iot_transfer":
        sdf, sy, slabel = load_csv(paths[src_name])
        dev, ydev, ldev = load_csv(paths["iot_dev"])
        test, ytest, ltest = load_csv(paths["iot_test"])
        tr, va, _ = split_source(sdf, sy, seed)

        scols, names = resolve_canonical_columns(sdf, src_name, paths["mapping"])
        dcols, _ = resolve_canonical_columns(dev, "CICIoT2023", paths["mapping"])
        tcols, _ = resolve_canonical_columns(test, "CICIoT2023", paths["mapping"])
        xtr, med = clean_numeric(sdf.iloc[tr], scols)
        xva, _ = clean_numeric(sdf.iloc[va], scols, med)
        def mapped_matrix(df, cols):
            x = df.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            for j, c in enumerate(cols):
                x[c] = x[c].fillna(float(med.iloc[j]))
            return x.to_numpy(np.float32)
        xadapt = mapped_matrix(dev, dcols)
        xte = mapped_matrix(test, tcols)
        return ScenarioData(name, src_name, tgt_name, xtr, sy[tr], xva, sy[va], xte, ytest, xadapt, names, "gmm_logit")

    raise KeyError(name)


def predict_scores(model, x: np.ndarray) -> np.ndarray:
    p = model.predict_proba(x)
    if p.ndim == 2:
        return p[:, 1].astype(float)
    return np.asarray(p).ravel().astype(float)


def run_one(scenario: str, seed: int, paths: Dict[str, Path], out_dir: Path) -> List[Dict]:
    data = prepare_scenario(scenario, seed, paths)
    # Protocol sanity checks from the frozen v0.5.0 evidence: E2 uses 14 aligned
    # related-CIC features and E3 uses exactly seven verified canonical features.
    if scenario.startswith("E2_") and len(data.feature_names) != 14:
        raise RuntimeError(
            f"{scenario}: expected 14 frozen E2 features, got {len(data.feature_names)}. "
            "Do not report this run; reuse the live repository's E2 feature-alignment utility."
        )
    if scenario.startswith("E3_") and len(data.feature_names) != 7:
        raise RuntimeError(
            f"{scenario}: expected 7 verified canonical features, got {len(data.feature_names)}."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    models = make_models(seed)
    rows = []

    for model_name, model in models.items():
        print(f"[E7] scenario={scenario} seed={seed} model={model_name}", flush=True)
        t0 = time.time()
        sw = compute_sample_weight(class_weight="balanced", y=data.y_train)
        if model_name == "CatBoost":
            model.fit(data.x_train, data.y_train, sample_weight=sw, eval_set=(data.x_val, data.y_val), early_stopping_rounds=50, verbose=False)
        else:
            model.fit(data.x_train, data.y_train, sample_weight=sw)
        train_seconds = time.time() - t0

        val_scores = predict_scores(model, data.x_val)
        source_thr, source_val_macro_f1 = best_macro_f1_threshold(data.y_val, val_scores)
        threshold = source_thr
        gmm_status = "not_used"
        target_adapt_pos = float("nan")
        if data.threshold_strategy == "gmm_logit" and data.x_target_adapt is not None:
            adapt_scores = predict_scores(model, data.x_target_adapt)
            threshold, gmm_status = gmm_logit_threshold(adapt_scores, seed)
            target_adapt_pos = float(np.mean(adapt_scores >= threshold))

        test_scores = predict_scores(model, data.x_test)
        m = metrics_from_scores(data.y_test, test_scores, threshold)
        row = {
            "paper_stage": "E7",
            "scenario": scenario,
            "seed": seed,
            "model": model_name,
            "train_domain": data.train_domain,
            "test_domain": data.test_domain,
            "feature_count": len(data.feature_names),
            "feature_names": json.dumps(data.feature_names),
            "target_labels_in_training": False,
            "source_val_decision_threshold": source_thr,
            "source_val_macro_f1": source_val_macro_f1,
            "decision_threshold": threshold,
            "target_threshold_strategy": data.threshold_strategy,
            "threshold_status": gmm_status,
            "target_adapt_positive_rate_at_threshold": target_adapt_pos,
            "train_seconds": train_seconds,
            **m,
        }
        rows.append(row)
        print(json.dumps(row, allow_nan=True), flush=True)

        # Save raw score bundle so E8 can compare operating points without retraining.
        score_path = out_dir / f"scores_{scenario}_{model_name}_s{seed}.npz"
        np.savez_compressed(
            score_path,
            y_test=data.y_test.astype(np.int8),
            test_scores=test_scores.astype(np.float32),
            source_val_threshold=np.array([source_thr], dtype=np.float32),
            decision_threshold=np.array([threshold], dtype=np.float32),
        )

    metrics_path = out_dir / "metrics.csv"
    new = pd.DataFrame(rows)
    if metrics_path.exists():
        old = pd.read_csv(metrics_path)
        old = old[~((old["scenario"] == scenario) & (old["seed"] == seed) & old["model"].isin(new["model"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(metrics_path, index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS.keys()])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-2017", type=Path, default=DEFAULT_PATHS["2017"])
    ap.add_argument("--data-2018", type=Path, default=DEFAULT_PATHS["2018"])
    ap.add_argument("--iot-dev", type=Path, default=DEFAULT_PATHS["iot_dev"])
    ap.add_argument("--iot-test", type=Path, default=DEFAULT_PATHS["iot_test"])
    ap.add_argument("--mapping", type=Path, default=DEFAULT_PATHS["mapping"])
    ap.add_argument("--output-dir", type=Path, default=Path("results/paper_E7_strong_tabular"))
    args = ap.parse_args()

    paths = {
        "2017": args.data_2017,
        "2018": args.data_2018,
        "iot_dev": args.iot_dev,
        "iot_test": args.iot_test,
        "mapping": args.mapping,
    }
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    all_rows = []
    for s in scenarios:
        all_rows.extend(run_one(s, args.seed, paths, args.output_dir))
    print(f"[E7] wrote {len(all_rows)} rows; aggregate with build_e7_table.py")


if __name__ == "__main__":
    main()
