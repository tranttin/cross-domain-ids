#!/usr/bin/env python
from __future__ import annotations
import _bootstrap  # noqa: F401
import argparse
from pathlib import Path
import sys
import yaml

from ids.data.canonical_mapping import load_mapping_config
from ids.robustness.constraints import load_constraint_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/paper_experiments_v050.yaml')
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    ok = True
    print('IDS PAPER PROTOCOL GATE v0.5.2')
    print('=' * 64)
    print('python:', sys.executable)
    try:
        import tensorflow as tf
        tfp = str(Path(tf.__file__).resolve())
        print('tensorflow:', tfp)
        forbidden = cfg['protocol_gates'].get('forbidden_tensorflow_path_fragment')
        if args.final and forbidden and forbidden.lower() in tfp.lower():
            print('FAIL: TensorFlow is loaded from forbidden environment fragment:', forbidden)
            ok = False
    except Exception as e:
        print('FAIL: TensorFlow import:', repr(e)); ok = False

    for key in ['CICIDS2017', 'CICIDS2018']:
        p = Path(cfg['datasets'][key]['csv'])
        print(('OK  ' if p.exists() else 'MISS'), key, p)
        ok &= p.exists()
    if args.final:
        for key in ['CICIoT2023_DEV', 'CICIoT2023_FINAL_TEST']:
            p = Path(cfg['datasets'][key]['csv'])
            print(('OK  ' if p.exists() else 'MISS'), key, p)
            ok &= p.exists()
        mapping = load_mapping_config(cfg['protocol_gates']['canonical_mapping'])

        # Strict canonical core used for heterogeneous transfer.
        # These mappings must be semantically verified.
        core_features = {
            'flow_packets_per_second',
            'forward_packets_per_second',
            'backward_packets_per_second',
            'packet_length_min',
            'packet_length_max',
            'packet_length_mean',
            'packet_length_std',
        }

        # Mappings are retained for documentation/future analysis but are
        # deliberately excluded from the strict core because their semantics
        # are not sufficiently compatible across CICFlowMeter and CICIoT2023.
        excluded_from_core = {
            'flow_duration',
            'fin_flag_count',
            'syn_flag_count',
            'rst_flag_count',
            'ack_flag_count',
            'urg_flag_count',
        }

        unverified_core = []

        for cname, entry in mapping.get('features', {}).items():
            m = (entry or {}).get('mappings', {}).get('CICIoT2023')
            if not m:
                continue

            if cname in core_features and not bool(m.get('verified', False)):
                unverified_core.append(cname)

        print(
            'OK  excluded_from_core (not blocking):',
            ', '.join(sorted(excluded_from_core))
        )

        if unverified_core:
            print(
                'FAIL: unverified CICIoT2023 core mappings:',
                ', '.join(unverified_core)
            )
            ok = False
        else:
            print('OK  canonical mapping verified (core set)')

        constraint_path = Path(cfg['paper'].get(
            'e6_constraint_config', 'configs/robustness/e6_constraints_v051.yaml'
        ))
        if not constraint_path.exists():
            print('FAIL: missing E6 constraint config:', constraint_path)
            ok = False
        else:
            constraints = load_constraint_config(constraint_path)
            mutable = set(constraints.get('mutable_semantic_features', []))
            if mutable != core_features:
                print('FAIL: E6 mutable features differ from strict canonical core')
                print('      expected:', ', '.join(sorted(core_features)))
                print('      actual  :', ', '.join(sorted(mutable)))
                ok = False
            elif not bool((constraints.get('fgsm') or {}).get('projected')):
                print('FAIL: E6 FGSM projection is not enabled')
                ok = False
            elif not bool((constraints.get('pgd') or {}).get('projected')):
                print('FAIL: E6 PGD projection is not enabled')
                ok = False
            elif set((constraints.get('adversarial_training') or {}).get('enabled_variants', [])) != {'mlp_at','mlp_dann_at'}:
                print('FAIL: E6 adversarial-training variants are incomplete')
                ok = False
            elif str(constraints.get('attack_scope')) != 'malicious_only':
                print('FAIL: E6 attack_scope must be malicious_only')
                ok = False
            elif [float(x) for x in constraints.get('epsilons', [])] != [float(x) for x in cfg['paper']['attack_epsilons']]:
                print('FAIL: E6 epsilon grids differ between paper matrix and constraint policy')
                ok = False
            else:
                print('OK  E6 constraint policy frozen:', constraints.get('protocol'))
    else:
        print('NOTE: development/smoke mode does not require frozen final IoT files or verified mapping.')

    print('\nSTATUS:', 'PASS' if ok else 'FAIL')
    raise SystemExit(0 if ok else 2)

if __name__ == '__main__':
    main()
