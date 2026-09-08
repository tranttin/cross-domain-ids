#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from ids.data.ciciot2023 import prepare_ciciot2023


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/ciciot2023")
    ap.add_argument("--output", default="data/processed/ciciot2023_sample.csv")
    ap.add_argument("--sample-frac", type=float, default=0.01, help="Fraction sampled from each CSV chunk; 0.01 = about 1%.")
    ap.add_argument("--task", choices=["binary", "category", "fine"], default="binary")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-files", type=int)
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.rglob("*.csv"))
    if args.max_files is not None:
        files = files[:args.max_files]
    out = prepare_ciciot2023(
        args.raw_dir, args.output, sample_frac=args.sample_frac,
        seed=args.seed, task=args.task, max_files=args.max_files,
    )
    meta = {
        "raw_dir": str(raw_dir.resolve()), "input_file_count": len(files),
        "input_files": [str(x) for x in files], "sample_frac": args.sample_frac,
        "seed": args.seed, "task": args.task, "output": str(Path(args.output).resolve()),
        "output_rows": int(len(out)), "output_columns": int(out.shape[1]),
        "label_counts": {str(k): int(v) for k, v in out["Label"].value_counts().to_dict().items()},
    }
    meta_path = Path(args.output).with_suffix(Path(args.output).suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved:", Path(args.output).resolve())
    print("Metadata:", meta_path.resolve())
    print("Shape:", out.shape)
    print(out["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
