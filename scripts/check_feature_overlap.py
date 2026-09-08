#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path
import pandas as pd

from ids.data.feature_schema import feature_overlap
from ids.data.canonical_mapping import resolve_mapping, mapping_dataframe


def columns_only(path: str):
    return pd.read_csv(path, nrows=0).columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--source-name")
    ap.add_argument("--target-name")
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--include-unverified", action="store_true")
    ap.add_argument("--output", default="results/feature_overlap.csv")
    args = ap.parse_args()
    scols, tcols = columns_only(args.source), columns_only(args.target)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)

    if args.source_name and args.target_name:
        rm = resolve_mapping(
            scols, tcols, args.source_name, args.target_name,
            mapping_path=args.mapping, verified_only=not args.include_unverified,
        )
        df = mapping_dataframe(rm)
        df.to_csv(out, index=False)
        print(f"Explicit mapped features: {len(rm.features)} (version={rm.version})")
        print(df.to_string(index=False) if len(df) else "<none>")
        print(f"Missing source entries: {len(rm.missing_source)}; missing target entries: {len(rm.missing_target)}")
        print(f"Unverified candidates skipped/flagged: {len(rm.unverified)}")
        if args.include_unverified:
            print("WARNING: unverified mappings are diagnostic only.")
    else:
        ov = feature_overlap(scols, tcols)
        rows = [{"canonical": k, "source_column": ov.source_map[k], "target_column": ov.target_map[k]} for k in ov.common]
        df = pd.DataFrame(rows)
        df.to_csv(out, index=False)
        print(f"Formatting-only common features: {len(ov.common)}")
        print(df.to_string(index=False) if len(df) else "<none>")
        print(f"Source-only: {len(ov.source_only)}; target-only: {len(ov.target_only)}")
        print("NOTE: formatting-only overlap does NOT establish semantic equivalence.")
    print("Saved:", out)


if __name__ == "__main__":
    main()
