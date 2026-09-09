from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


_EPS = 1e-7


def _as_prob2(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected probability matrix with shape (n, 2); got {arr.shape}")
    arr = np.clip(arr, _EPS, 1.0 - _EPS)
    arr = arr / np.sum(arr, axis=1, keepdims=True)
    return arr


@dataclass
class DomainHeadAudit:
    source_n: int
    target_n: int
    source_domain_label: int
    target_domain_label: int
    pooled_accuracy: float
    balanced_accuracy: float
    reversed_balanced_accuracy: float
    domain_auroc: float
    reversed_domain_auroc: float
    domain_log_loss: float
    source_accuracy: float
    target_accuracy: float
    source_mean_p_target: float
    target_mean_p_target: float
    source_median_p_target: float
    target_median_p_target: float
    p_target_gap_target_minus_source: float
    source_predicted_as_target_rate: float
    target_predicted_as_target_rate: float
    orientation_status: str

    def to_dict(self) -> dict:
        return asdict(self)


def audit_domain_head_outputs(source_prob: np.ndarray, target_prob: np.ndarray) -> DomainHeadAudit:
    """Audit DANN domain-head outputs with fixed domain semantics.

    Clean-DANN uses source-domain label 0 and target-domain label 1.  This function
    never consumes target *class* labels.  It separates three cases that can look
    similar in a single pooled accuracy number: normal domain separation, reversed
    head orientation, and near-random/confused domain predictions.
    """
    ps = _as_prob2(source_prob)
    pt = _as_prob2(target_prob)
    ys = np.zeros(len(ps), dtype=int)
    yt = np.ones(len(pt), dtype=int)
    y = np.concatenate([ys, yt])
    prob = np.concatenate([ps, pt], axis=0)
    pred = np.argmax(prob, axis=1)
    score = prob[:, 1]

    src_pred = np.argmax(ps, axis=1)
    tgt_pred = np.argmax(pt, axis=1)
    src_acc = float(np.mean(src_pred == 0)) if len(src_pred) else float("nan")
    tgt_acc = float(np.mean(tgt_pred == 1)) if len(tgt_pred) else float("nan")
    bal = 0.5 * (src_acc + tgt_acc)
    rev_bal = 0.5 * (float(np.mean(src_pred == 1)) + float(np.mean(tgt_pred == 0)))
    pooled = float(np.mean(pred == y))
    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
    rev_auc = float(1.0 - auc) if np.isfinite(auc) else float("nan")
    ce = float(log_loss(y, prob, labels=[0, 1]))

    s_mean = float(np.mean(ps[:, 1]))
    t_mean = float(np.mean(pt[:, 1]))
    gap = t_mean - s_mean

    if auc >= 0.70:
        status = "normal_separation"
    elif auc <= 0.30:
        status = "reversed_separation"
    elif 0.45 <= auc <= 0.55:
        status = "confused_or_aligned"
    else:
        status = "weak_separation"

    return DomainHeadAudit(
        source_n=int(len(ps)),
        target_n=int(len(pt)),
        source_domain_label=0,
        target_domain_label=1,
        pooled_accuracy=pooled,
        balanced_accuracy=float(bal),
        reversed_balanced_accuracy=float(rev_bal),
        domain_auroc=auc,
        reversed_domain_auroc=rev_auc,
        domain_log_loss=ce,
        source_accuracy=src_acc,
        target_accuracy=tgt_acc,
        source_mean_p_target=s_mean,
        target_mean_p_target=t_mean,
        source_median_p_target=float(np.median(ps[:, 1])),
        target_median_p_target=float(np.median(pt[:, 1])),
        p_target_gap_target_minus_source=float(gap),
        source_predicted_as_target_rate=float(np.mean(src_pred == 1)),
        target_predicted_as_target_rate=float(np.mean(tgt_pred == 1)),
        orientation_status=status,
    )


@dataclass
class DomainProbeAudit:
    n_per_domain: int
    test_fraction: float
    balanced_accuracy: float
    domain_auroc: float
    reversed_domain_auroc: float
    coefficient_norm: float

    def to_dict(self) -> dict:
        return asdict(self)


def frozen_domain_probe(
    source_representation: np.ndarray,
    target_representation: np.ndarray,
    *,
    seed: int = 42,
    max_per_domain: int = 10000,
    test_fraction: float = 0.5,
) -> DomainProbeAudit:
    """Fit an orientation-independent post-hoc domain probe on frozen representations.

    Only domain membership labels (source=0, target=1) are used.  Target attack/benign
    labels are not read.  High AUROC means the frozen representation still exposes
    domain information; AUROC near 0.5 means domains are hard to separate.
    """
    xs = np.asarray(source_representation, dtype=float)
    xt = np.asarray(target_representation, dtype=float)
    if xs.ndim != 2 or xt.ndim != 2 or xs.shape[1] != xt.shape[1]:
        raise ValueError(f"Representations must be 2-D with same width; got {xs.shape}, {xt.shape}")
    n = min(len(xs), len(xt), int(max_per_domain))
    if n < 20:
        raise ValueError("Need at least 20 samples per domain for a stable audit probe")
    rng = np.random.default_rng(seed)
    si = rng.choice(len(xs), size=n, replace=False)
    ti = rng.choice(len(xt), size=n, replace=False)
    X = np.vstack([xs[si], xt[ti]])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=float(test_fraction), random_state=seed, stratify=y
    )
    probe = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    probe.fit(Xtr, ytr)
    score = probe.predict_proba(Xte)[:, list(probe.classes_).index(1)]
    pred = (score >= 0.5).astype(int)
    auc = float(roc_auc_score(yte, score))
    bal = float(balanced_accuracy_score(yte, pred))
    return DomainProbeAudit(
        n_per_domain=int(n),
        test_fraction=float(test_fraction),
        balanced_accuracy=bal,
        domain_auroc=auc,
        reversed_domain_auroc=float(1.0 - auc),
        coefficient_norm=float(np.linalg.norm(probe.coef_)),
    )
