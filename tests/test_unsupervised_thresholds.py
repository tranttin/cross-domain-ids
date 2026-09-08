import numpy as np

from ids.evaluation.unsupervised_thresholds import (
    choose_unsupervised_target_threshold,
    gmm_logit_threshold,
    source_prior_quantile_threshold,
    valley_logit_threshold,
)


def test_gmm_logit_finds_two_shifted_score_groups_without_labels():
    rng = np.random.default_rng(42)
    benign = np.clip(rng.normal(0.001, 0.0002, 500), 1e-6, 0.01)
    attack = np.clip(rng.normal(0.03, 0.004, 4500), 0.005, 0.08)
    score = np.concatenate([benign, attack])
    r = gmm_logit_threshold(score, seed=42)
    assert 0.001 < r.threshold < 0.03
    assert 0.80 < r.target_adapt_positive_rate < 0.99
    assert r.details["uses_target_labels"] is False


def test_valley_logit_finds_large_gap():
    score = np.concatenate([np.linspace(0.001, 0.003, 100), np.linspace(0.03, 0.05, 900)])
    r = valley_logit_threshold(score)
    assert 0.003 < r.threshold < 0.03
    assert 0.85 < r.target_adapt_positive_rate < 0.95


def test_source_prior_quantile_matches_source_prevalence_approximately():
    score = np.linspace(0.0, 1.0, 1001)
    r = source_prior_quantile_threshold(score, source_positive_prevalence=0.8)
    assert abs(r.target_adapt_positive_rate - 0.8) < 0.01


def test_dispatch_source_strategy_preserves_source_threshold():
    score = np.linspace(0.01, 0.99, 100)
    r = choose_unsupervised_target_threshold(
        "source",
        target_adapt_score=score,
        source_threshold=0.42,
        source_positive_prevalence=0.7,
        seed=42,
    )
    assert abs(r.threshold - 0.42) < 1e-12
    assert r.strategy == "source"
