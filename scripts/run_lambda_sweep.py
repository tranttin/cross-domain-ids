#!/usr/bin/env python
"""Sequential clean-DANN lambda sweep with resume support.

Example:
  python scripts/run_lambda_sweep.py --direction 2017-2018 --lambdas 0.1 0.5 1.0

Each lambda is logged into results/summary.csv by run_cross_domain_clean.py.
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["both", "2017-2018", "2018-2017"], default="both")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.1, 0.5, 1.0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="configs/clean_cross_domain.yaml")
    ap.add_argument("--domain-loss-weight", type=float, default=0.5)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    for lam in args.lambdas:
        cmd = [
            sys.executable, "scripts/run_cross_domain_clean.py",
            "--config", args.config,
            "--direction", args.direction,
            "--only", "dann",
            "--seed", str(args.seed),
            "--lambda-d", str(lam),
            "--domain-loss-weight", str(args.domain_loss_weight),
        ]
        if not args.no_resume:
            cmd.append("--resume")
        print("\n=== lambda_d =", lam, "===")
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nLambda sweep completed. Inspect results/summary.csv or aggregate_results.py.")


if __name__ == "__main__":
    main()
