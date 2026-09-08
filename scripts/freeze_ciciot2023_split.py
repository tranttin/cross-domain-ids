#!/usr/bin/env python
"""Freeze a fresh prepared CICIoT2023 pool into DEV and untouched FINAL_TEST files."""
from __future__ import annotations
import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def sha256(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Fresh prepared CICIoT2023 pool; do not reuse the tuning diagnostic file.')
    ap.add_argument('--dev-output', default='data/processed/ciciot2023_dev.csv')
    ap.add_argument('--test-output', default='data/processed/ciciot2023_final_test.csv')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=20260819)
    ap.add_argument('--force', action='store_true')
    args=ap.parse_args()
    inp=Path(args.input); dev=Path(args.dev_output); test=Path(args.test_output)
    if (dev.exists() or test.exists()) and not args.force:
        raise SystemExit('Refusing to overwrite frozen split. Use --force only before any final-paper run.')
    df=pd.read_csv(inp, low_memory=False)
    if 'Label' not in df.columns: raise SystemExit('Input must contain normalized Label')
    y=(df['Label'].astype(int)!=0).astype(int)
    d,t=train_test_split(df, test_size=args.test_size, random_state=args.seed, stratify=y)
    dev.parent.mkdir(parents=True,exist_ok=True); test.parent.mkdir(parents=True,exist_ok=True)
    d.reset_index(drop=True).to_csv(dev,index=False); t.reset_index(drop=True).to_csv(test,index=False)
    manifest={
        'protocol':'v0.5.0_frozen_ciciot2023_holdout', 'seed':args.seed,
        'source_input':str(inp.resolve()), 'source_rows':len(df),
        'dev_file':str(dev.resolve()), 'dev_rows':len(d), 'dev_sha256':sha256(dev),
        'final_test_file':str(test.resolve()), 'final_test_rows':len(t), 'final_test_sha256':sha256(test),
        'final_test_policy':'Never use labels or scores from FINAL_TEST for feature selection, threshold selection, lambda selection, checkpointing, or debugging.',
    }
    mp=dev.parent/'ciciot2023_frozen_split_manifest.json'
    mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
