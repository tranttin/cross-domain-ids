from pathlib import Path
import yaml


def _cfg():
    p = Path(__file__).parents[1] / 'configs' / 'features' / 'canonical_features.yaml'
    return yaml.safe_load(p.read_text(encoding='utf-8'))


def test_flow_duration_does_not_use_ttl_duration():
    cfg = _cfg()
    m = cfg['features']['flow_duration']['mappings']['CICIoT2023']
    assert m['column'] == 'flow_duration'


def test_flag_counts_use_count_fields():
    cfg = _cfg()
    for canonical, target in [
        ('fin_flag_count','fin_count'),
        ('syn_flag_count','syn_count'),
        ('rst_flag_count','rst_count'),
        ('ack_flag_count','ack_count'),
        ('urg_flag_count','urg_count'),
    ]:
        assert cfg['features'][canonical]['mappings']['CICIoT2023']['column'] == target


def test_protocol_not_in_numeric_mapping():
    cfg = _cfg()
    assert 'protocol' not in cfg['features']
