import pandas as pd
from ids.data.domain_diagnostics import diagnose_aligned_frames


def test_domain_diagnostics_detects_shift_and_classes():
    s = pd.DataFrame({"x": list(range(20)), "y": [0] * 20, "Label": [0, 1] * 10})
    t = pd.DataFrame({"x": list(range(100, 120)), "y": [0] * 20, "Label": [1] * 18 + [0] * 2})
    d = diagnose_aligned_frames(s, t, "source", "target", seed=42)
    row = d.out_of_range.set_index("feature").loc["x"]
    assert row["target_outside_source_minmax"] == 1.0
    target_attack = d.class_distribution[(d.class_distribution.dataset == "target") & (d.class_distribution.label == 1)].iloc[0]
    assert abs(target_attack["fraction"] - 0.9) < 1e-9
    assert bool(d.missing_or_invalid.set_index("feature").loc["y", "target_constant"])
