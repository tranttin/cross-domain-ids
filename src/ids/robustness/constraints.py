from __future__ import annotations

"""Semantic feasibility projection for the final E6 robustness protocol.

The attack budget is measured in the normalized model-input space.  Only the
seven frozen canonical continuous variables may change.  Projection also keeps
the perturbation inside the per-sample L-infinity ball, prevents motion farther
outside the source-training range, and enforces the raw-space packet-length
relations declared in ``e6_constraints_v051.yaml``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from ids.data.canonical_mapping import load_mapping_config
from ids.data.feature_schema import canonical_feature_name


TOL = 1e-6


def load_constraint_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    required = {"protocol", "mutable_semantic_features", "raw_rules"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"Constraint config {path} is missing: {missing}")
    return cfg


def _as_2d(x: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(x, dtype=np.float32)
    had_channel = arr.ndim == 3 and arr.shape[-1] == 1
    if had_channel:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected [n, features] or [n, features, 1], got {arr.shape}")
    return arr, had_channel


def _restore_shape(x: np.ndarray, had_channel: bool) -> np.ndarray:
    return x[..., None] if had_channel else x


def _dataset_column_for_semantic(
    mapping_cfg: dict[str, Any], semantic: str, dataset_name: str
) -> str | None:
    entry = (mapping_cfg.get("features") or {}).get(semantic) or {}
    mappings = entry.get("mappings") or {}
    value = mappings.get(dataset_name)
    if value is None:
        lower = {str(k).lower(): v for k, v in mappings.items()}
        value = lower.get(str(dataset_name).lower())
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("column")
    return None if value is None else str(value)


def resolve_mutable_coordinates(
    selected_feature_names: list[str],
    mutable_semantic_features: list[str],
    *,
    dataset_name: str,
    canonical_mapping_path: str | Path,
) -> dict[str, int]:
    """Resolve every mutable semantic variable to exactly one model coordinate."""
    lookup: dict[str, list[int]] = {}
    for i, name in enumerate(selected_feature_names):
        lookup.setdefault(canonical_feature_name(name), []).append(i)

    mapping_cfg = load_mapping_config(canonical_mapping_path)
    resolved: dict[str, int] = {}
    for semantic in mutable_semantic_features:
        candidates = [semantic]
        dataset_column = _dataset_column_for_semantic(mapping_cfg, semantic, dataset_name)
        if dataset_column:
            candidates.append(dataset_column)
        matches: set[int] = set()
        for candidate in candidates:
            matches.update(lookup.get(canonical_feature_name(candidate), []))
        if len(matches) != 1:
            raise ValueError(
                f"Mutable semantic feature {semantic!r} resolved to {sorted(matches)} "
                f"in selected features {selected_feature_names}; exactly one coordinate is required"
            )
        resolved[str(semantic)] = next(iter(matches))
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(f"Mutable semantic features do not map one-to-one: {resolved}")
    return resolved


def _pava_three(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Vectorized equal-weight isotonic projection for three ordered values."""
    a, b, c = a.copy(), b.copy(), c.copy()
    bad = a > b
    mean = 0.5 * (a + b)
    a = np.where(bad, mean, a)
    b = np.where(bad, mean, b)
    bad = b > c
    mean = 0.5 * (b + c)
    b = np.where(bad, mean, b)
    c = np.where(bad, mean, c)
    bad = a > b
    mean = (a + b + c) / 3.0
    a = np.where(bad, mean, a)
    b = np.where(bad, mean, b)
    c = np.where(bad, mean, c)
    return a, b, c


@dataclass
class SemanticConstraintProjector:
    selected_feature_names: list[str]
    semantic_to_index: dict[str, int]
    center: np.ndarray
    scale: np.ndarray
    source_min: np.ndarray
    source_max: np.ndarray
    nonnegative_semantics: tuple[str, ...]
    ordered_triplet: tuple[str, str, str] | None
    protocol: str

    @property
    def mutable_indices(self) -> np.ndarray:
        return np.asarray(sorted(self.semantic_to_index.values()), dtype=np.int64)

    @property
    def mutable_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.selected_feature_names), dtype=bool)
        mask[self.mutable_indices] = True
        return mask

    def _scaled_to_raw(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale[None, :] + self.center[None, :]

    def _raw_to_scaled(self, x: np.ndarray) -> np.ndarray:
        return (x - self.center[None, :]) / self.scale[None, :]

    def project(self, candidate, reference, epsilon: float):
        x, had_channel = _as_2d(candidate)
        x0, ref_had_channel = _as_2d(reference)
        if had_channel != ref_had_channel or x.shape != x0.shape:
            raise ValueError(f"Candidate/reference shape mismatch: {np.shape(candidate)} vs {np.shape(reference)}")
        if x.shape[1] != len(self.selected_feature_names):
            raise ValueError("Attack input width does not match the frozen feature order")
        eps = float(epsilon)
        if eps < 0:
            raise ValueError("epsilon must be non-negative")

        # The clean target can lie outside the source range.  Expanding the box to
        # include x0 freezes outward motion without silently modifying the clean row.
        lo = np.maximum(x0 - eps, self.source_min[None, :])
        hi = np.minimum(x0 + eps, self.source_max[None, :])
        lo = np.minimum(lo, x0)
        hi = np.maximum(hi, x0)

        out = np.clip(x, lo, hi)
        out[:, ~self.mutable_mask] = x0[:, ~self.mutable_mask]

        raw = self._scaled_to_raw(out)
        raw0 = self._scaled_to_raw(x0)
        raw_lo = self._scaled_to_raw(lo)
        raw_hi = self._scaled_to_raw(hi)
        for semantic in self.nonnegative_semantics:
            idx = self.semantic_to_index[semantic]
            clean_invalid = raw0[:, idx] < -TOL
            valid = ~clean_invalid
            raw_lo[valid, idx] = np.maximum(raw_lo[valid, idx], 0.0)
            if np.any(raw_lo[valid, idx] > raw_hi[valid, idx] + TOL):
                raise ValueError(
                    f"No feasible point exists for non-negative constraint {semantic!r}"
                )
            raw[valid, idx] = np.clip(raw[valid, idx], raw_lo[valid, idx], raw_hi[valid, idx])
            # Never silently repair a dirty clean row: freeze that coordinate and
            # audit it as a pre-existing dataset inconsistency.
            raw[clean_invalid, idx] = raw0[clean_invalid, idx]

        if self.ordered_triplet is not None:
            s_min, s_mean, s_max = self.ordered_triplet
            i_min = self.semantic_to_index[s_min]
            i_mean = self.semantic_to_index[s_mean]
            i_max = self.semantic_to_index[s_max]

            clean_invalid = (
                (raw0[:, i_min] > raw0[:, i_mean] + TOL) |
                (raw0[:, i_mean] > raw0[:, i_max] + TOL)
            )
            valid = ~clean_invalid

            # Existing dataset inconsistencies cannot be made feasible at
            # epsilon=0 without changing the clean test set. Freeze the three
            # coordinates for those rows and require zero attack-induced change.
            for idx in (i_min, i_mean, i_max):
                raw[clean_invalid, idx] = raw0[clean_invalid, idx]

            # Propagate lower/upper boxes so a monotone triple is guaranteed to exist.
            l_min = raw_lo[valid, i_min]
            l_mean = np.maximum(raw_lo[valid, i_mean], l_min)
            l_max = np.maximum(raw_lo[valid, i_max], l_mean)
            u_max = raw_hi[valid, i_max]
            u_mean = np.minimum(raw_hi[valid, i_mean], u_max)
            u_min = np.minimum(raw_hi[valid, i_min], u_mean)
            if np.any(l_min > u_min + TOL) or np.any(l_mean > u_mean + TOL) or np.any(l_max > u_max + TOL):
                raise ValueError(
                    "No feasible packet_length_min <= packet_length_mean <= packet_length_max "
                    "point exists inside the perturbation box; audit the clean input"
                )

            a, b, c = _pava_three(raw[valid, i_min], raw[valid, i_mean], raw[valid, i_max])
            a = np.clip(a, l_min, u_min)
            b = np.clip(b, np.maximum(l_mean, a), u_mean)
            c = np.clip(c, np.maximum(l_max, b), u_max)
            raw[valid, i_min], raw[valid, i_mean], raw[valid, i_max] = a, b, c

        out = self._raw_to_scaled(raw)
        out = np.clip(out, lo, hi)
        out[:, ~self.mutable_mask] = x0[:, ~self.mutable_mask]
        return _restore_shape(out.astype(np.float32), had_channel)

    def audit(self, candidate, reference, epsilon: float) -> dict[str, float | int | bool]:
        x, _ = _as_2d(candidate)
        x0, _ = _as_2d(reference)
        delta = x - x0
        raw = self._scaled_to_raw(x)
        raw0 = self._scaled_to_raw(x0)
        imm = ~self.mutable_mask
        immutable_max = float(np.max(np.abs(delta[:, imm]))) if np.any(imm) and len(x) else 0.0
        linf = float(np.max(np.abs(delta))) if len(x) else 0.0
        nonnegative_violations = 0
        absolute_nonnegative_violations = 0
        clean_nonnegative_violations = 0
        preexisting_nonnegative_modified = 0
        for semantic in self.nonnegative_semantics:
            idx = self.semantic_to_index[semantic]
            current_bad = raw[:, idx] < -TOL
            clean_bad = raw0[:, idx] < -TOL
            absolute_nonnegative_violations += int(np.sum(current_bad))
            clean_nonnegative_violations += int(np.sum(clean_bad))
            nonnegative_violations += int(np.sum(current_bad & ~clean_bad))
            preexisting_nonnegative_modified += int(
                np.sum(clean_bad & (np.abs(raw[:, idx] - raw0[:, idx]) > 5 * TOL))
            )
        order_violations = 0
        absolute_order_violations = 0
        clean_order_violations = 0
        preexisting_order_modified = 0
        if self.ordered_triplet is not None:
            a, b, c = (self.semantic_to_index[s] for s in self.ordered_triplet)
            current_bad = (raw[:, a] > raw[:, b] + TOL) | (raw[:, b] > raw[:, c] + TOL)
            clean_bad = (raw0[:, a] > raw0[:, b] + TOL) | (raw0[:, b] > raw0[:, c] + TOL)
            absolute_order_violations = int(np.sum(current_bad))
            clean_order_violations = int(np.sum(clean_bad))
            order_violations = int(np.sum(current_bad & ~clean_bad))
            changed = np.max(np.abs(raw[:, [a, b, c]] - raw0[:, [a, b, c]]), axis=1) > 5 * TOL
            preexisting_order_modified = int(np.sum(clean_bad & changed))
        finite_violations = int(np.size(x) - np.count_nonzero(np.isfinite(x)))
        epsilon_violations = int(np.sum(np.max(np.abs(delta), axis=1) > float(epsilon) + 5 * TOL))
        immutable_violations = int(np.sum(np.abs(delta[:, imm]) > 5 * TOL)) if np.any(imm) else 0
        effective_lo = np.maximum(x0 - float(epsilon), self.source_min[None, :])
        effective_hi = np.minimum(x0 + float(epsilon), self.source_max[None, :])
        effective_lo = np.minimum(effective_lo, x0)
        effective_hi = np.maximum(effective_hi, x0)
        source_range_violations = int(np.sum((x < effective_lo - 5 * TOL) | (x > effective_hi + 5 * TOL)))
        total = (
            finite_violations + epsilon_violations + immutable_violations +
            nonnegative_violations + order_violations + source_range_violations +
            preexisting_nonnegative_modified + preexisting_order_modified
        )
        return {
            "constraint_violation_count": int(total),
            "constraint_pass": bool(total == 0),
            "finite_violation_count": finite_violations,
            "epsilon_violation_count": epsilon_violations,
            "immutable_violation_count": immutable_violations,
            "nonnegative_violation_count": nonnegative_violations,
            "ordered_triplet_violation_count": order_violations,
            "absolute_nonnegative_violation_count": absolute_nonnegative_violations,
            "clean_nonnegative_violation_count": clean_nonnegative_violations,
            "preexisting_nonnegative_modified_count": preexisting_nonnegative_modified,
            "absolute_ordered_triplet_violation_count": absolute_order_violations,
            "clean_ordered_triplet_violation_count": clean_order_violations,
            "preexisting_ordered_triplet_modified_count": preexisting_order_modified,
            "source_range_outward_violation_count": source_range_violations,
            "immutable_max_abs_delta": immutable_max,
            "max_linf_delta": linf,
        }

    def assert_clean_feasible(self, x) -> dict[str, float | int | bool]:
        audit = self.audit(x, x, 0.0)
        if not audit["constraint_pass"]:
            raise ValueError(f"Clean evaluation data violate frozen semantic constraints: {audit}")
        return audit


def build_semantic_projector(
    *,
    selected_feature_names: list[str],
    scaler,
    source_train_scaled,
    dataset_name: str,
    constraint_path: str | Path = "configs/robustness/e6_constraints_v051.yaml",
    canonical_mapping_path: str | Path = "configs/features/canonical_features.yaml",
) -> tuple[SemanticConstraintProjector, dict[str, Any]]:
    cfg = load_constraint_config(constraint_path)
    mutable = [str(x) for x in cfg["mutable_semantic_features"]]
    resolved = resolve_mutable_coordinates(
        selected_feature_names,
        mutable,
        dataset_name=dataset_name,
        canonical_mapping_path=canonical_mapping_path,
    )
    center = np.asarray(scaler.center_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("RobustScaler has invalid/non-positive scales")
    source, _ = _as_2d(source_train_scaled)
    if source.shape[1] != len(selected_feature_names):
        raise ValueError("Source train width and selected feature list differ")
    raw_rules = cfg.get("raw_rules") or {}
    triplet = raw_rules.get("ordered_triplet")
    ordered = None
    if triplet:
        ordered = (str(triplet["min"]), str(triplet["mean"]), str(triplet["max"]))
    projector = SemanticConstraintProjector(
        selected_feature_names=list(selected_feature_names),
        semantic_to_index=resolved,
        center=center,
        scale=scale,
        source_min=np.min(source, axis=0).astype(np.float32),
        source_max=np.max(source, axis=0).astype(np.float32),
        nonnegative_semantics=tuple(str(x) for x in raw_rules.get("nonnegative", [])),
        ordered_triplet=ordered,
        protocol=str(cfg["protocol"]),
    )
    return projector, cfg


ProjectFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]
