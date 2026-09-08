#!/usr/bin/env python
"""Run the v0.4.8 staged BatchNorm-frozen DANN control for lambda=0 and lambda=0.1.

This wrapper intentionally launches two independent clean-runner jobs. Each job
source-pretrains the same architecture before adaptation. Use smoke mode first.
"""
from __future__ import annotations
import argparse, subprocess, sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-csv', required=True)
    ap.add_argument('--source-name', required=True)
    ap.add_argument('--target-csv', required=True)
    ap.add_argument('--target-name', required=True)
    ap.add_argument('--allow-unverified-mapping', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--target-threshold-strategy', default='gmm_logit', choices=['source','gmm_logit','valley_logit','source_prior_quantile'])
    ap.add_argument('--lambdas', nargs='+', type=float, default=[0.0,0.1])
    args=ap.parse_args()
    base=[sys.executable,'scripts/run_transfer_clean.py',
          '--source-csv',args.source_csv,'--source-name',args.source_name,
          '--target-csv',args.target_csv,'--target-name',args.target_name,
          '--encoder','mlp_bn_fair','--only','all',
          '--target-threshold-strategy',args.target_threshold_strategy]
    if args.allow_unverified_mapping: base.append('--allow-unverified-mapping')
    if args.smoke: base.append('--smoke')
    for lam in args.lambdas:
        cmd=base+['--lambda-d',str(lam)]
        print('\n=== staged DANN control lambda_d=',lam,'===', flush=True)
        subprocess.run(cmd, check=True)

if __name__=='__main__': main()
