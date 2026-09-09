from __future__ import annotations

from pathlib import Path
from typing import Any

from ids.utils.config import load_yaml


DEFAULT_REGISTRY = Path("configs/datasets.yaml")


def load_dataset_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, dict[str, Any]]:
    cfg = load_yaml(path)
    datasets = cfg.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError("configs/datasets.yaml must contain a 'datasets' mapping")
    return datasets


def get_dataset_spec(name: str, path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    datasets = load_dataset_registry(path)
    if name not in datasets:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(sorted(datasets))}")
    return datasets[name]
