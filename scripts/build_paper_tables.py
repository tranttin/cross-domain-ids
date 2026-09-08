#!/usr/bin/env python
"""Build paper-facing CSV tables from results/summary.csv without hand copying."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ['accuracy','macro_f1','balanced_accuracy','auprc','auroc','fpr','fnr']


def latest_per(df, keys):
    if df.empty: return df
    if 'run_id' in df.columns:
        return df.sort_values('run_id').drop_duplicates(keys, keep='last')
    return df.drop_duplicates(keys, keep='last')


def aggregate(df, group_cols, metric_names=None):
    metric_names = METRICS if metric_names is None else metric_names
    metrics = [m for m in metric_names if m in df.columns]
    if df.empty:
        return df
    out = df.groupby(group_cols, dropna=False)[metrics].agg(['mean','std','count']).reset_index()
    out.columns = ['_'.join([x for x in c if x]) if isinstance(c, tuple) else c for c in out.columns]
    return out


def save(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print('\n###', path)
    print(df.to_string(index=False) if len(df) else '(no rows yet)')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--summary', default='results/summary.csv')
    ap.add_argument('--output-dir', default='results/paper_tables')
    ap.add_argument('--lambda-main', type=float, default=0.01)
    args=ap.parse_args()
    df=pd.read_csv(args.summary)
    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # T2: clean in-domain benchmark.
    t2=df[df.get('paper_table', pd.Series('',index=df.index)).astype(str).eq('T2_InDomain')].copy()
    t2=latest_per(t2, ['seed','model','train_domain','test_domain'])
    save(aggregate(t2, ['model','test_domain']), out/'T2_in_domain_mean_std.csv')

    # T3: related CIC transfer. Keep only paper methods.
    related=df[
        df['train_domain'].astype(str).isin(['2017','2018']) &
        df['test_domain'].astype(str).isin(['2017','2018']) &
        (df['train_domain'].astype(str) != df['test_domain'].astype(str))
    ].copy()
    related=related[related['model'].astype(str).isin(['LogisticRegression','LogReg','RandomForest','CNN-BiLSTM-Attention','Clean-DANN'])]
    if 'lambda_d' in related.columns:
        is_dann=related['model'].astype(str).eq('Clean-DANN')
        lam=pd.to_numeric(related['lambda_d'],errors='coerce')
        related=related[(~is_dann) | np.isclose(lam,args.lambda_main,equal_nan=False)]
    related=latest_per(related,['seed','model','train_domain','test_domain'])
    save(aggregate(related,['model','train_domain','test_domain']), out/'T3_CIC_cross_domain_mean_std.csv')

    # T4: heterogeneous transfer to IoT.
    hetero=df[
        df['train_domain'].astype(str).isin(['2017','2018']) &
        df['test_domain'].astype(str).eq('CICIoT2023')
    ].copy()
    wanted=['LogisticRegression','RandomForest','MLP-BN-Fair-Canonical-Control','MLP-BN-Fair-DANN-Canonical-Control']
    hetero=hetero[hetero['model'].astype(str).isin(wanted)]
    if 'lambda_d' in hetero.columns:
        is_dann=hetero['model'].astype(str).eq('MLP-BN-Fair-DANN-Canonical-Control')
        lam=pd.to_numeric(hetero['lambda_d'],errors='coerce')
        hetero=hetero[(~is_dann) | np.isclose(lam,args.lambda_main,equal_nan=False)]
    # Prefer final/verified rows if present; otherwise diagnostics still show for development.
    if 'mapping_verified_only' in hetero.columns and hetero['mapping_verified_only'].fillna(False).any():
        hetero=hetero[hetero['mapping_verified_only'].fillna(False)]
    hetero=latest_per(hetero,['seed','model','train_domain','test_domain'])
    save(aggregate(hetero,['model','train_domain','test_domain']), out/'T4_heterogeneous_transfer_mean_std.csv')

    # T5: multi-source.
    t5=df[df.get('paper_table', pd.Series('',index=df.index)).astype(str).eq('T5_MultiSource')].copy()
    if 'lambda_d' in t5.columns:
        is_dann=t5['adaptation'].astype(str).eq('UDA') if 'adaptation' in t5.columns else pd.Series(False,index=t5.index)
        lam=pd.to_numeric(t5['lambda_d'],errors='coerce')
        t5=t5[(~is_dann) | np.isclose(lam,args.lambda_main,equal_nan=False)]
    t5=latest_per(t5,['seed','model','train_domain','test_domain'])
    save(aggregate(t5,['model','train_domain','test_domain']), out/'T5_multisource_mean_std.csv')

    # F3: lambda sensitivity: development-only, latest seed/lambda diagnostic.
    ls=df[
        df['model'].astype(str).eq('MLP-BN-Fair-DANN-Canonical-Control') &
        df['train_domain'].astype(str).eq('2018') &
        df['test_domain'].astype(str).eq('CICIoT2023')
    ].copy()
    if 'lambda_d' in ls.columns:
        ls['lambda_d']=pd.to_numeric(ls['lambda_d'],errors='coerce')
        ls=ls[ls['lambda_d'].notna()]
    ls=latest_per(ls,['seed','lambda_d'])
    cols=[c for c in ['seed','lambda_d','best_source_val_macro_f1','final_val_domain_accuracy',*METRICS,'decision_threshold'] if c in ls.columns]
    save(ls[cols].sort_values(['seed','lambda_d']), out/'F3_lambda_sensitivity_rows.csv')

    # T6/F4: final constraint-aware robustness evidence.
    t6=df[df.get('paper_table', pd.Series('',index=df.index)).astype(str).eq('T6_Robustness')].copy()
    t6=latest_per(t6,[c for c in ['seed','scenario','model','attack','epsilon'] if c in t6.columns])
    save(t6.sort_values([c for c in ['scenario','model','attack','epsilon','seed'] if c in t6.columns]), out/'E6_robustness_rows.csv')
    e6_metrics = [
        'clean_macro_f1','robust_macro_f1','asr','attack_success_rate',
        'fnr_adv','clean_fnr','constraint_violation_count','max_linf_delta',
    ]
    save(
        aggregate(t6,[c for c in ['scenario','model','attack','epsilon'] if c in t6.columns],e6_metrics),
        out/'T6_robustness_mean_std.csv',
    )

if __name__=='__main__':
    main()
