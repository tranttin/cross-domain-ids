from __future__ import annotations

"""Label-free target-threshold adaptation for binary cross-domain evaluation.

All functions in this module select a threshold from TARGET *unlabelled adaptation*
scores only, optionally using SOURCE-labelled statistics such as source prevalence.
They must never inspect target-test labels. The chosen threshold can then be frozen
and evaluated on a held-out target test set.
"""

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture


EPS = 1e-8


@dataclass(frozen=True)
class UnsupervisedThresholdResult:
    strategy: str
    threshold: float
    status: str
    target_adapt_positive_rate: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["threshold"] = float(out["threshold"])
        out["target_adapt_positive_rate"] = float(out["target_adapt_positive_rate"])
        return out


def _score_array(score) -> np.ndarray:
    s = np.asarray(score, dtype=float).reshape(-1)
    if len(s) == 0:
        raise ValueError("Target adaptation score array is empty")
    if not np.all(np.isfinite(s)):
        raise ValueError("Target adaptation scores contain NaN/Inf")
    return np.clip(s, EPS, 1.0 - EPS)


def _logit(s: np.ndarray) -> np.ndarray:
    return np.log(s / (1.0 - s))


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x = np.asarray(x, dtype=float)
    out = 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
    return float(out) if out.ndim == 0 else out


def _result(strategy: str, threshold: float, score: np.ndarray, status: str, **details) -> UnsupervisedThresholdResult:
    threshold = float(np.clip(threshold, EPS, 1.0 - EPS))
    return UnsupervisedThresholdResult(
        strategy=strategy,
        threshold=threshold,
        status=status,
        target_adapt_positive_rate=float(np.mean(score >= threshold)),
        details={k: (v.item() if isinstance(v, np.generic) else v) for k, v in details.items()},
    )


def fixed_source_threshold(source_threshold: float, target_adapt_score) -> UnsupervisedThresholdResult:
    s = _score_array(target_adapt_score)
    return _result("source", float(source_threshold), s, "ok", uses_target_labels=False)


def source_prior_quantile_threshold(
    target_adapt_score,
    *,
    source_positive_prevalence: float,
) -> UnsupervisedThresholdResult:
    """Choose threshold so target-adapt predicted-positive rate matches SOURCE prevalence.

    This is a deliberately simple label-free control. It assumes class-prior stability,
    which may be badly violated under domain shift; therefore it should not be treated
    as the preferred calibration method without evidence.
    """
    s = _score_array(target_adapt_score)
    p = float(np.clip(source_positive_prevalence, 0.0, 1.0))
    if p <= 0.0:
        t = np.nextafter(float(np.max(s)), np.inf)
    elif p >= 1.0:
        t = np.nextafter(float(np.min(s)), -np.inf)
    else:
        t = float(np.quantile(s, 1.0 - p))
    return _result(
        "source_prior_quantile", t, s, "ok",
        uses_target_labels=False,
        source_positive_prevalence=p,
        assumption="target class prior approximately matches labelled source train prior",
    )


def _gmm_intersection_threshold(means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> float:
    """Return posterior-equality threshold between two 1D Gaussian components.

    A dense grid is used instead of a fragile closed-form quadratic because we only
    have two one-dimensional components and threshold selection is a diagnostic step.
    """
    lo = float(min(means[0] - 5 * stds[0], means[1] - 5 * stds[1]))
    hi = float(max(means[0] + 5 * stds[0], means[1] + 5 * stds[1]))
    grid = np.linspace(lo, hi, 20001)
    dens = []
    for m, sd, w in zip(means, stds, weights):
        sd = max(float(sd), 1e-6)
        z = (grid - float(m)) / sd
        dens.append(float(w) * np.exp(-0.5 * z * z) / sd)
    diff = dens[0] - dens[1]
    between = (grid >= min(means)) & (grid <= max(means))
    idxs = np.where(between)[0]
    if len(idxs) == 0:
        return float(np.mean(means))
    local = idxs[np.argmin(np.abs(diff[idxs]))]
    return float(grid[local])


def gmm_logit_threshold(
    target_adapt_score,
    *,
    seed: int = 42,
    min_component_weight: float = 0.005,
    min_mean_separation: float = 0.25,
) -> UnsupervisedThresholdResult:
    """Fit a 2-component GMM to target-adaptation logit scores, label-free."""
    s = _score_array(target_adapt_score)
    z = _logit(s).reshape(-1, 1)
    if np.std(z) < 1e-8:
        return _result("gmm_logit", float(np.median(s)), s, "degenerate_scores", uses_target_labels=False)

    gm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=10,
        random_state=seed,
        reg_covar=1e-6,
    )
    gm.fit(z)
    means = gm.means_.reshape(-1)
    stds = np.sqrt(gm.covariances_.reshape(-1))
    weights = gm.weights_.reshape(-1)
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]

    separation = float(abs(means[1] - means[0]))
    status = "ok"
    if float(np.min(weights)) < min_component_weight:
        status = "tiny_component"
    elif separation < min_mean_separation:
        status = "weak_separation"

    zt = _gmm_intersection_threshold(means, stds, weights)
    t = _sigmoid(zt)
    return _result(
        "gmm_logit", t, s, status,
        uses_target_labels=False,
        component_logit_means=[float(x) for x in means],
        component_logit_stds=[float(x) for x in stds],
        component_weights=[float(x) for x in weights],
        logit_mean_separation=separation,
    )


def valley_logit_threshold(
    target_adapt_score,
    *,
    trim_fraction: float = 0.01,
    min_cluster_fraction: float = 0.01,
) -> UnsupervisedThresholdResult:
    """Threshold at the largest robust gap in sorted target-adaptation logit scores.

    Extreme tails are trimmed and candidate gaps must leave at least
    ``min_cluster_fraction`` samples on each side. This is label-free and makes no
    class-prior assumption, but it is a heuristic rather than a calibrated estimator.
    """
    s = _score_array(target_adapt_score)
    z = np.sort(_logit(s))
    n = len(z)
    if n < 20 or np.std(z) < 1e-8:
        return _result("valley_logit", float(np.median(s)), s, "insufficient_structure", uses_target_labels=False)

    trim = int(max(0, np.floor(trim_fraction * n)))
    min_side = int(max(2, np.ceil(min_cluster_fraction * n)))
    lo = max(trim, min_side)
    hi = min(n - trim - 1, n - min_side - 1)
    if hi <= lo:
        return _result("valley_logit", float(np.median(s)), s, "insufficient_structure", uses_target_labels=False)

    gaps = np.diff(z)
    candidate = np.arange(lo, hi + 1)
    idx = int(candidate[np.argmax(gaps[candidate])])
    zt = float((z[idx] + z[idx + 1]) / 2.0)
    t = _sigmoid(zt)
    med_gap = float(np.median(gaps[candidate])) if len(candidate) else 0.0
    gap = float(gaps[idx])
    ratio = float(gap / max(med_gap, 1e-12))
    status = "ok" if ratio >= 3.0 else "weak_gap"
    return _result(
        "valley_logit", t, s, status,
        uses_target_labels=False,
        largest_logit_gap=gap,
        gap_to_median_gap_ratio=ratio,
        lower_cluster_fraction=float((idx + 1) / n),
        upper_cluster_fraction=float(1.0 - (idx + 1) / n),
    )


def choose_unsupervised_target_threshold(
    strategy: str,
    *,
    target_adapt_score,
    source_threshold: float,
    source_positive_prevalence: float,
    seed: int = 42,
) -> UnsupervisedThresholdResult:
    strategy = str(strategy).lower()
    if strategy == "source":
        return fixed_source_threshold(source_threshold, target_adapt_score)
    if strategy == "gmm_logit":
        return gmm_logit_threshold(target_adapt_score, seed=seed)
    if strategy == "valley_logit":
        return valley_logit_threshold(target_adapt_score)
    if strategy == "source_prior_quantile":
        return source_prior_quantile_threshold(
            target_adapt_score,
            source_positive_prevalence=source_positive_prevalence,
        )
    raise ValueError(f"Unknown target threshold strategy: {strategy}")
