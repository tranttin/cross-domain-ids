#!/usr/bin/env python
"""Audit source/target semantics and distribution BEFORE a deep transfer run.

This script is intentionally allowed to read target labels for OFFLINE DIAGNOSTICS.
Those labels are never passed to UDA training/model selection. The report labels target
class counts as evaluation-only to prevent accidental protocol confusion.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd

from ids.data.loaders import load_csv
from ids.data.canonical_mapping import (
    resolve_mapping, align_frames_with_mapping, mapping_dataframe,
)
from ids.data.domain_diagnostics import diagnose_aligned_frames, scaled_shift_table
from ids.preprocessing.cross_domain import prepare_clean_source_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--target-csv", required=True)
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--output", help="Default: results/diagnostics/<source>_to_<target>")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-unverified-mapping", action="store_true", help="Include candidate mappings in diagnostics only.")
    ap.add_argument("--variance-threshold", type=float, default=0.01)
    args = ap.parse_args()

    out = Path(args.output or f"results/diagnostics/{args.source_name}_to_{args.target_name}")
    out.mkdir(parents=True, exist_ok=True)
    source = load_csv(args.source_csv)
    target = load_csv(args.target_csv)

    mapping = resolve_mapping(
        source.columns, target.columns, args.source_name, args.target_name,
        mapping_path=args.mapping, verified_only=not args.allow_unverified_mapping,
    )
    mapping_dataframe(mapping).to_csv(out / "feature_mapping.csv", index=False)
    (out / "missing_source.txt").write_text("\n".join(mapping.missing_source), encoding="utf-8")
    (out / "missing_target.txt").write_text("\n".join(mapping.missing_target), encoding="utf-8")
    (out / "unverified_candidates.txt").write_text("\n".join(mapping.unverified), encoding="utf-8")

    if not mapping.features:
        raise SystemExit(
            "No usable mapped features. Inspect feature_mapping/missing files and verify mappings in "
            f"{args.mapping}. For candidate-only diagnostics, rerun with --allow-unverified-mapping."
        )

    src_aligned, tgt_aligned = align_frames_with_mapping(source, target, mapping)
    tables = diagnose_aligned_frames(src_aligned, tgt_aligned, args.source_name, args.target_name, seed=args.seed)
    tables.feature_stats.to_csv(out / "feature_stats.csv", index=False)
    tables.class_distribution.to_csv(out / "class_distribution.csv", index=False)
    tables.out_of_range.to_csv(out / "out_of_range.csv", index=False)
    tables.missing_or_invalid.to_csv(out / "missing_or_invalid.csv", index=False)

    split = prepare_clean_source_target(
        src_aligned, tgt_aligned, args.source_name, args.target_name,
        seed=args.seed, threshold=args.variance_threshold, feature_alignment="exact",
    )
    scaled = scaled_shift_table(split)
    scaled.to_csv(out / "scaled_shift.csv", index=False)

    suspicious = scaled.sort_values("target_outside_scaled_source_p01_p99", ascending=False).head(10)
    invalid = tables.missing_or_invalid[
        (tables.missing_or_invalid["target_constant"]) |
        (tables.missing_or_invalid["source_constant"]) |
        (tables.missing_or_invalid["target_missing_rate"] > 0.01) |
        (tables.missing_or_invalid["source_missing_rate"] > 0.01)
    ]
    class_rows = tables.class_distribution.to_dict(orient="records")
    report = [
        "IDS v0.4 DOMAIN-SHIFT DIAGNOSTIC",
        "=" * 42,
        f"source={args.source_name} target={args.target_name}",
        f"mapping_version={mapping.version}",
        f"mapped_features={len(mapping.features)} selected_after_variance={len(split.selected_feature_names)}",
        f"mapping_mode={'CANDIDATE/UNVERIFIED' if args.allow_unverified_mapping else 'VERIFIED_ONLY'}",
        "",
        "CLASS DISTRIBUTION (target labels are evaluation-only diagnostics):",
        *[f"  {r['dataset']} {r['name']}: n={r['count']} fraction={r['fraction']:.4f}" for r in class_rows],
        "",
        "TOP TARGET OUT-OF-SOURCE-RANGE FEATURES AFTER SOURCE-FITTED SCALING:",
        suspicious[["feature", "target_outside_scaled_source_p01_p99", "target_scaled_median", "source_scaled_p99"]].to_string(index=False),
        "",
        f"Features with >1% missing or constant behavior: {len(invalid)}",
        invalid.to_string(index=False) if len(invalid) else "  <none>",
        "",
        "INTERPRETATION:",
        "- High out-of-range rates suggest statistical/unit mismatch; inspect semantics before training.",
        "- Candidate/unverified mappings are diagnostic only and must not be used as final paper evidence.",
        "- A source-only LR/RF sanity transfer should be run next; DANN is not a substitute for bad feature semantics.",
    ]
    (out / "diagnostic_report.txt").write_text("\n".join(report), encoding="utf-8")
    json.dump({
        "source": args.source_name, "target": args.target_name,
        "mapping_version": mapping.version, "mapped_features": len(mapping.features),
        "selected_features": split.selected_feature_names,
        "verified_only": not args.allow_unverified_mapping,
    }, (out / "diagnostic_manifest.json").open("w", encoding="utf-8"), indent=2)

    print("\n".join(report[:9]))
    print("\nTop shifted features:\n", suspicious.head(8).to_string(index=False))
    print("\nSaved diagnostics to:", out.resolve())


if __name__ == "__main__":
    main()
