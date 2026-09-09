from __future__ import annotations

import re
from dataclasses import dataclass
import pandas as pd


def canonical_feature_name(name: str) -> str:
    """Conservative name normalization for cross-dataset audits.

    This only resolves formatting differences (case, spaces, punctuation). It does
    NOT claim semantic equivalence between abbreviations or differently defined flow
    statistics.
    """
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


@dataclass
class FeatureOverlap:
    common: list[str]
    source_map: dict[str, str]
    target_map: dict[str, str]
    source_only: list[str]
    target_only: list[str]


def _canonical_map(columns) -> dict[str, str]:
    out: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for raw in columns:
        if raw in {"Label", "Label_bin"}:
            continue
        key = canonical_feature_name(raw)
        if key in out and out[key] != raw:
            duplicates.setdefault(key, [out[key]]).append(raw)
        else:
            out[key] = raw
    if duplicates:
        raise ValueError(f"Ambiguous columns after canonicalization: {duplicates}")
    return out


def feature_overlap(source_columns, target_columns) -> FeatureOverlap:
    sm = _canonical_map(source_columns)
    tm = _canonical_map(target_columns)
    common = sorted(set(sm) & set(tm))
    return FeatureOverlap(
        common=common,
        source_map={k: sm[k] for k in common},
        target_map={k: tm[k] for k in common},
        source_only=sorted(set(sm) - set(tm)),
        target_only=sorted(set(tm) - set(sm)),
    )
