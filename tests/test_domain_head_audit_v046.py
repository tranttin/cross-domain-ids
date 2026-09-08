import numpy as np
from ids.evaluation.domain_head import audit_domain_head_outputs, frozen_domain_probe


def test_domain_head_detects_reversed_orientation():
    # Source is wrongly assigned high P(target), target low P(target).
    source = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
    target = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.9, 0.1]])
    a = audit_domain_head_outputs(source, target)
    assert a.orientation_status == "reversed_separation"
    assert a.balanced_accuracy < 0.1
    assert a.reversed_balanced_accuracy > 0.9
    assert a.domain_auroc < 0.1
    assert a.reversed_domain_auroc > 0.9


def test_domain_head_detects_confusion():
    source = np.array([[0.51, 0.49], [0.49, 0.51], [0.50, 0.50], [0.52, 0.48]])
    target = np.array([[0.50, 0.50], [0.48, 0.52], [0.51, 0.49], [0.49, 0.51]])
    a = audit_domain_head_outputs(source, target)
    assert 0.25 <= a.domain_auroc <= 0.75


def test_frozen_domain_probe_finds_separable_representation():
    rng = np.random.default_rng(42)
    source = rng.normal(loc=-2.0, scale=0.5, size=(200, 4))
    target = rng.normal(loc=2.0, scale=0.5, size=(200, 4))
    p = frozen_domain_probe(source, target, seed=42, max_per_domain=200)
    assert p.domain_auroc > 0.95
    assert p.balanced_accuracy > 0.9
