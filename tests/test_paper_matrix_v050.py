from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from ids.preprocessing.in_domain import preprocess_clean_binary_in_domain_fixed_test
from ids.preprocessing.cross_domain import prepare_clean_source_target_fixed_test


def _frame(n=100, seed=1):
    rng=np.random.default_rng(seed)
    y=np.array([0,1]*(n//2),dtype=int)
    return pd.DataFrame({
        'f1': rng.normal(size=n) + y,
        'f2': rng.normal(size=n),
        'Label': y,
    })


def test_fixed_in_domain_keeps_external_test_size():
    dev=_frame(100,1); test=_frame(40,2)
    s=preprocess_clean_binary_in_domain_fixed_test(dev,test,'CICIoT2023',seed=42,variance_threshold=0.0)
    assert s.X_test.shape[0] == 40
    assert len(s.y_test) == 40
    assert s.X_train.shape[1] == s.X_test.shape[1]


def test_fixed_cross_domain_uses_all_target_adapt_and_test():
    src=_frame(120,3); adapt=_frame(50,4); test=_frame(30,5)
    s=prepare_clean_source_target_fixed_test(src,adapt,test,'2018','CICIoT2023',seed=42,threshold=0.0,feature_alignment='exact')
    assert s.X_target_unlabeled.shape[0] == 50
    assert s.X_target_test.shape[0] == 30
    assert len(s.y_target_test) == 30


def test_paper_yaml_has_expected_stages_and_lambda():
    cfg=yaml.safe_load(Path('configs/paper_experiments_v050.yaml').read_text())
    assert set(cfg['stages']) == {'E0','E1','E2','E3','E4','E5','E6'}
    assert cfg['paper']['main_dann_lambda'] == 0.01
    assert cfg['paper']['final_seeds'] == [42,43,44]
    assert cfg['paper']['e6_constraint_config'] == 'configs/robustness/e6_constraints_v051.yaml'
    assert cfg['stages']['E6']['status'] == 'constraint_aware_final_enabled'
