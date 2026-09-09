from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ids.data.feature_schema import canonical_feature_name


@dataclass(frozen=True)
class ResolvedFeature:
    canonical: str
    source_column: str
    target_column: str
    verified: bool
    source_verified: bool
    target_verified: bool
    note: str = ""


@dataclass
class ResolvedMapping:
    version: str
    source_name: str
    target_name: str
    features: list[ResolvedFeature]
    missing_source: list[str]
    missing_target: list[str]
    unverified: list[str]

    @property
    def verified_features(self) -> list[ResolvedFeature]:
        return [f for f in self.features if f.verified]


def load_mapping_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "features" not in cfg:
        raise ValueError(f"Mapping file {path} has no 'features' section")
    return cfg


def _column_lookup(columns) -> dict[str, str]:
    lookup: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for raw in columns:
        key = canonical_feature_name(raw)
        if key in lookup and lookup[key] != raw:
            duplicates.setdefault(key, [lookup[key]]).append(raw)
        else:
            lookup[key] = raw
    if duplicates:
        raise ValueError(f"Ambiguous dataset columns after formatting normalization: {duplicates}")
    return lookup


def _dataset_mapping(entry: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    mappings = entry.get("mappings", {}) or {}
    # Exact key first, then case-insensitive key. Do not guess aliases here.
    if dataset_name in mappings:
        value = mappings[dataset_name]
        return value if isinstance(value, dict) else {"column": value}
    lower = {str(k).lower(): v for k, v in mappings.items()}
    value = lower.get(str(dataset_name).lower())
    if value is None:
        return None
    return value if isinstance(value, dict) else {"column": value}


def resolve_mapping(
    source_columns,
    target_columns,
    source_name: str,
    target_name: str,
    *,
    mapping_path: str | Path = "configs/features/canonical_features.yaml",
    verified_only: bool = True,
) -> ResolvedMapping:
    cfg = load_mapping_config(mapping_path)
    source_lookup = _column_lookup(source_columns)
    target_lookup = _column_lookup(target_columns)

    resolved: list[ResolvedFeature] = []
    missing_source: list[str] = []
    missing_target: list[str] = []
    unverified: list[str] = []

    for canonical, entry in (cfg.get("features") or {}).items():
        entry = entry or {}
        sm = _dataset_mapping(entry, source_name)
        tm = _dataset_mapping(entry, target_name)
        if sm is None or tm is None:
            continue

        sraw = str(sm.get("column", "")).strip()
        traw = str(tm.get("column", "")).strip()
        if not sraw or not traw:
            continue
        scol = source_lookup.get(canonical_feature_name(sraw))
        tcol = target_lookup.get(canonical_feature_name(traw))
        if scol is None:
            missing_source.append(f"{canonical}: {sraw}")
        if tcol is None:
            missing_target.append(f"{canonical}: {traw}")
        if scol is None or tcol is None:
            continue

        sv = bool(sm.get("verified", False))
        tv = bool(tm.get("verified", False))
        verified = sv and tv
        note_parts = [str(entry.get("description", "")).strip(), str(tm.get("note", "")).strip()]
        note = " | ".join(p for p in note_parts if p)
        feature = ResolvedFeature(
            canonical=str(canonical), source_column=scol, target_column=tcol,
            verified=verified, source_verified=sv, target_verified=tv, note=note,
        )
        if not verified:
            unverified.append(str(canonical))
            if verified_only:
                continue
        resolved.append(feature)

    return ResolvedMapping(
        version=str(cfg.get("version", "unknown")),
        source_name=source_name,
        target_name=target_name,
        features=resolved,
        missing_source=missing_source,
        missing_target=missing_target,
        unverified=unverified,
    )


def align_frames_with_mapping(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    mapping: ResolvedMapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two frames with identical canonical feature names + original Label.

    The function only renames/reorders columns already resolved by an explicit mapping.
    It never invents values, converts units, or fills semantically missing features.
    """
    if not mapping.features:
        raise ValueError("Resolved mapping contains zero usable features")
    if "Label" not in source_df.columns or "Label" not in target_df.columns:
        raise KeyError("Both prepared datasets must contain a normalized 'Label' column")

    canonical = [f.canonical for f in mapping.features]
    source = source_df[[f.source_column for f in mapping.features]].copy()
    target = target_df[[f.target_column for f in mapping.features]].copy()
    source.columns = canonical
    target.columns = canonical
    source["Label"] = source_df["Label"].to_numpy()
    target["Label"] = target_df["Label"].to_numpy()
    return source, target


def mapping_dataframe(mapping: ResolvedMapping) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "canonical": f.canonical,
            "source_column": f.source_column,
            "target_column": f.target_column,
            "source_verified": f.source_verified,
            "target_verified": f.target_verified,
            "verified": f.verified,
            "note": f.note,
            "mapping_version": mapping.version,
        }
        for f in mapping.features
    ])
