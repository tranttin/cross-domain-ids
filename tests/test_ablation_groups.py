from pathlib import Path
import yaml


def test_v042_ablation_groups_quarantine_suspect_flags():
    cfg = yaml.safe_load(Path("configs/features/ablation_groups.yaml").read_text(encoding="utf-8"))
    groups = cfg["groups"]
    assert "rst_flag_count" in groups["no_rst_urg"]["exclude"]
    assert "urg_flag_count" in groups["no_rst_urg"]["exclude"]
    core = groups["rates_lengths_core"]["include"]
    assert "flow_duration" in core
    assert "packet_length_mean" in core
    assert "rst_flag_count" not in core
