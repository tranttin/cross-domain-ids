#!/usr/bin/env python
"""Run a fixed experiment command over multiple seeds sequentially.

Example:
  python scripts/run_seed_sweep.py --experiment clean_cross_domain --seeds 42 43 44 -- --direction 2017-2018

Everything after `--` is forwarded to the selected experiment script.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPTS = {
    "baselines_2018": "scripts/run_baselines_2018.py",
    "hybrid_2018": "scripts/run_hybrid_2018.py",
    "legacy_cross_domain": "scripts/run_cross_domain_legacy.py",
    "clean_cross_domain": "scripts/run_cross_domain_clean.py",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=SCRIPTS, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    for seed in args.seeds:
        cmd = [sys.executable, SCRIPTS[args.experiment], "--seed", str(seed), *extra]
        print("\n=== seed", seed, "===")
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)
    print("\nSweep completed. Aggregate with: python scripts/aggregate_results.py")


if __name__ == "__main__":
    main()
