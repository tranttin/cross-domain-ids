from ids.data.registry import load_dataset_registry


def test_required_datasets_registered():
    r = load_dataset_registry()
    for name in ["cse_cic_ids2018_processed", "cic_ids2017_processed", "ciciot2023", "unsw_nb15", "iot23"]:
        assert name in r
