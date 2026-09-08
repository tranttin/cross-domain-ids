#!/usr/bin/env python
"""Single entry point for the paper experiment matrix (E6 final enabled).

Examples
--------
# See exactly what will run
python scripts/run_paper.py --stage E3 --mode smoke --dry-run

# Execute E1 smoke
python scripts/run_paper.py --stage E1 --mode smoke

# Execute final E3 for seeds from YAML (requires protocol gate PASS)
python scripts/run_paper.py --stage E3 --mode final
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def q(x):
    return str(x)


def dataset(cfg, key):
    return cfg['datasets'][key]


def add_final_gate(cmds, config):
    return [[sys.executable, 'scripts/check_paper_protocol.py', '--config', config, '--final'], *cmds]


def e1_commands(cfg, mode, seeds):
    cmds = []
    for seed in seeds:
        # 2017 + 2018 use their prepared complete files.
        for key in ['CICIDS2017', 'CICIDS2018']:
            d = dataset(cfg, key)
            c = [sys.executable, 'scripts/run_in_domain_clean.py', '--csv', d['csv'], '--dataset-name', d['name'], '--seed', q(seed)]
            if mode == 'smoke': c += ['--smoke']
            cmds.append(c)
        # IoT final uses frozen DEV and FINAL_TEST; smoke stays on diagnostic sample.
        if mode == 'final':
            dev = dataset(cfg, 'CICIoT2023_DEV'); test = dataset(cfg, 'CICIoT2023_FINAL_TEST')
            cmds.append([sys.executable, 'scripts/run_in_domain_clean.py', '--csv', dev['csv'], '--test-csv', test['csv'], '--dataset-name', dev['name'], '--seed', q(seed)])
        else:
            d = dataset(cfg, 'CICIoT2023_DIAGNOSTIC')
            cmds.append([sys.executable, 'scripts/run_in_domain_clean.py', '--csv', d['csv'], '--dataset-name', d['name'], '--seed', q(seed), '--smoke'])
    return cmds


def e2_commands(cfg, mode, seeds):
    cmds = []
    lam = cfg['paper']['main_dann_lambda']
    for seed in seeds:
        # Cheap LR/RF/SGD rows; paper table filters LR and RF.
        d17, d18 = dataset(cfg, 'CICIDS2017'), dataset(cfg, 'CICIDS2018')
        for src, tgt in [(d17, d18), (d18, d17)]:
            c = [sys.executable, 'scripts/run_transfer_baselines.py', '--source-csv', src['csv'], '--source-name', src['name'], '--target-csv', tgt['csv'], '--target-name', tgt['name'], '--seed', q(seed)]
            if mode == 'smoke': c += ['--max-source', '20000', '--max-target', '20000']
            cmds.append(c)
        # CNN source-only + clean DANN for both directions.
        c = [sys.executable, 'scripts/run_cross_domain_clean.py', '--direction', 'both', '--only', 'all', '--seed', q(seed), '--lambda-d', q(lam)]
        if mode == 'smoke': c += ['--smoke']
        cmds.append(c)
    return cmds


def _hetero_pair_commands(cfg, src_key, mode, seed, include_baselines=True):
    src = dataset(cfg, src_key)
    lam = cfg['paper']['main_dann_lambda']
    strategy = cfg['paper']['target_threshold_strategy']
    if mode == 'final':
        tgt = dataset(cfg, 'CICIoT2023_DEV'); final = dataset(cfg, 'CICIoT2023_FINAL_TEST')
        allow = []
        test_args = ['--target-test-csv', final['csv']]
    else:
        tgt = dataset(cfg, 'CICIoT2023_DIAGNOSTIC')
        allow = ['--allow-unverified-mapping']
        test_args = []
    cmds = []
    if include_baselines:
        c = [sys.executable, 'scripts/run_transfer_baselines.py', '--source-csv', src['csv'], '--source-name', src['name'], '--target-csv', tgt['csv'], '--target-name', tgt['name'], '--seed', q(seed), *test_args, *allow]
        if mode == 'smoke': c += ['--max-source', '20000', '--max-target', '20000']
        cmds.append(c)
    c = [sys.executable, 'scripts/run_transfer_clean.py', '--source-csv', src['csv'], '--source-name', src['name'], '--target-csv', tgt['csv'], '--target-name', tgt['name'], '--encoder', 'mlp_bn_fair', '--only', 'all', '--lambda-d', q(lam), '--target-threshold-strategy', strategy, '--seed', q(seed), *test_args, *allow]
    if mode == 'smoke': c += ['--smoke']
    cmds.append(c)
    return cmds


def e3_commands(cfg, mode, seeds):
    cmds = []
    for seed in seeds:
        cmds += _hetero_pair_commands(cfg, 'CICIDS2017', mode, seed)
        cmds += _hetero_pair_commands(cfg, 'CICIDS2018', mode, seed)
    return cmds


def e4_commands(cfg, mode, seeds):
    cmds = []
    a, b = dataset(cfg, 'CICIDS2017'), dataset(cfg, 'CICIDS2018')
    lam = cfg['paper']['main_dann_lambda']; strategy = cfg['paper']['target_threshold_strategy']
    for seed in seeds:
        if mode == 'final':
            tgt = dataset(cfg, 'CICIoT2023_DEV'); final = dataset(cfg, 'CICIoT2023_FINAL_TEST')
            tail = ['--target-test-csv', final['csv']]
        else:
            tgt = dataset(cfg, 'CICIoT2023_DIAGNOSTIC')
            tail = ['--allow-unverified-mapping', '--smoke']
        cmds.append([
            sys.executable, 'scripts/run_multisource_transfer.py',
            '--source-a-csv', a['csv'], '--source-a-name', a['name'],
            '--source-b-csv', b['csv'], '--source-b-name', b['name'],
            '--target-csv', tgt['csv'], '--target-name', tgt['name'],
            '--lambda-d', q(lam), '--target-threshold-strategy', strategy,
            '--seed', q(seed), *tail,
        ])
    return cmds


def e5_commands(cfg, mode, seeds):
    # E5 is deliberately a development ablation. Never point it at FINAL_TEST.
    cmds = []
    src = dataset(cfg, 'CICIDS2018'); tgt = dataset(cfg, 'CICIoT2023_DIAGNOSTIC')
    for seed in seeds:
        for lam in cfg['paper']['lambda_sweep']:
            cmds.append([
                sys.executable, 'scripts/run_transfer_clean.py',
                '--source-csv', src['csv'], '--source-name', src['name'],
                '--target-csv', tgt['csv'], '--target-name', tgt['name'],
                '--allow-unverified-mapping', '--encoder', 'mlp_bn_fair',
                '--only', 'dann', '--lambda-d', q(lam),
                '--target-threshold-strategy', cfg['paper']['target_threshold_strategy'],
                '--seed', q(seed), '--smoke',
            ])
    return cmds


def e6_commands(cfg, mode, seeds):
    cmds = []
    eps = [q(x) for x in cfg['paper']['attack_epsilons']]
    constraints = cfg['paper'].get('e6_constraint_config', 'configs/robustness/e6_constraints_v051.yaml')
    for seed in seeds:
        d18 = dataset(cfg, 'CICIDS2018')
        c1 = [
            sys.executable, 'scripts/run_robustness_eval.py',
            '--scenario', 'in_domain_2018', '--variants', 'cnn',
            '--source-csv', d18['csv'], '--source-name', d18['name'],
            '--constraint-config', constraints, '--seed', q(seed), '--eps', *eps,
        ]
        if mode == 'smoke': c1 += ['--smoke']
        cmds.append(c1)
        if mode == 'final':
            dev, final = dataset(cfg, 'CICIoT2023_DEV'), dataset(cfg, 'CICIoT2023_FINAL_TEST')
            tail = ['--target-test-csv', final['csv']]
        else:
            dev = dataset(cfg, 'CICIoT2023_DIAGNOSTIC')
            tail = ['--allow-unverified-mapping', '--smoke']
        cmds.append([
            sys.executable, 'scripts/run_robustness_eval.py', '--scenario', '2018_to_CICIoT2023',
            '--variants', 'mlp', 'mlp_dann', 'mlp_at', 'mlp_dann_at',
            '--source-csv', d18['csv'], '--source-name', d18['name'],
            '--target-csv', dev['csv'], '--target-name', dev['name'],
            '--constraint-config', constraints,
            '--lambda-d', q(cfg['paper']['main_dann_lambda']), '--seed', q(seed),
            '--eps', *eps, *tail,
        ])
    return cmds


BUILDERS = {'E1': e1_commands, 'E2': e2_commands, 'E3': e3_commands, 'E4': e4_commands, 'E5': e5_commands, 'E6': e6_commands}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/paper_experiments_v050.yaml')
    ap.add_argument('--stage', choices=['E0','E1','E2','E3','E4','E5','E6','all'], required=True)
    ap.add_argument('--mode', choices=['smoke','final'], default='smoke')
    ap.add_argument('--seed', type=int, action='append', help='Override YAML seeds; repeat option for multiple seeds.')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    seeds = args.seed or ([cfg['paper']['smoke_seed']] if args.mode == 'smoke' else cfg['paper']['final_seeds'])

    if args.stage == 'E0':
        cmds = [[sys.executable, 'scripts/check_paper_protocol.py', '--config', args.config] + (['--final'] if args.mode == 'final' else [])]
    else:
        stages = list(BUILDERS) if args.stage == 'all' else [args.stage]
        cmds = []
        for st in stages:
            cmds += BUILDERS[st](cfg, args.mode, seeds)
        if args.mode == 'final':
            cmds = add_final_gate(cmds, args.config)

    print(f'Paper matrix v0.5.2 | stage={args.stage} mode={args.mode} seeds={seeds}')
    print('=' * 84)
    for i, cmd in enumerate(cmds, 1):
        print(f'[{i:02d}] ' + ' '.join(map(str, cmd)))
    if args.dry_run:
        return
    for i, cmd in enumerate(cmds, 1):
        print(f'\n>>> RUN {i}/{len(cmds)}')
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
