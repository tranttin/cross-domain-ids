#!/usr/bin/env python
"""v0.4.6 DANN domain-head audit: orientation vs genuine domain separability.

This diagnostic retrains one small DANN control run and then audits:
1) the model's own domain-head orientation on source validation vs UNLABELLED target adaptation;
2) a post-hoc logistic domain probe on the raw canonical input;
3) the same probe on the frozen learned embedding.

No target attack/benign labels are used anywhere in the audit.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ids.data.canonical_mapping import align_frames_with_mapping, mapping_dataframe, resolve_mapping
from ids.data.loaders import load_csv
from ids.evaluation.domain_head import audit_domain_head_outputs, frozen_domain_probe
from ids.models.clean_dann import train_clean_dann
from ids.preprocessing.cross_domain import binary_labels_for_dataset, prepare_clean_source_target
from ids.utils.seed import set_global_seed


def _cap_source(df: pd.DataFrame, name: str, n: int | None, seed: int) -> pd.DataFrame:
    if not n or n >= len(df):
        return df
    y = binary_labels_for_dataset(df, name)
    rng = np.random.default_rng(seed)
    parts = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        k = max(1, int(round(n * len(idx) / len(df))))
        pick = rng.choice(idx, size=min(k, len(idx)), replace=False)
        parts.append(df.iloc[pick])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def _cap_target(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    # Unlabelled random cap: never stratify by target attack/benign labels.
    if not n or n >= len(df):
        return df
    return df.sample(n=int(n), random_state=seed).reset_index(drop=True)


def _embedding_model(model, encoder: str):
    import tensorflow as tf
    layer_name = "fair_embedding" if encoder == "mlp_fair" else ("mlp_embedding" if encoder == "mlp" else "gap_dann")
    try:
        layer = model.get_layer(layer_name)
    except ValueError as exc:
        raise RuntimeError(f"Cannot find embedding layer {layer_name!r} in model {model.name}") from exc
    return tf.keras.Model(model.input, layer.output, name=f"{model.name}_embedding_audit")


def _shape(x, encoder: str):
    arr = np.asarray(x, dtype=np.float32)
    return arr if encoder in {"mlp", "mlp_fair"} else arr[..., None]


def _interpret(head, input_probe, embedding_probe) -> str:
    if head.orientation_status == "reversed_separation" and head.reversed_domain_auroc >= 0.70:
        head_note = "domain head is strongly reversed at the audited checkpoint"
    elif head.orientation_status == "confused_or_aligned":
        head_note = "domain head is near random/confused"
    else:
        head_note = f"domain head status={head.orientation_status}"

    if embedding_probe.domain_auroc <= 0.60:
        rep_note = "frozen embedding is weakly domain-separable (alignment plausible)"
    elif embedding_probe.domain_auroc >= 0.80:
        rep_note = "frozen embedding remains strongly domain-separable (alignment is not achieved)"
    else:
        rep_note = "frozen embedding retains moderate domain information"

    if input_probe.domain_auroc >= 0.80 and embedding_probe.domain_auroc < input_probe.domain_auroc - 0.10:
        delta_note = "encoder reduced domain separability versus canonical input"
    elif embedding_probe.domain_auroc > input_probe.domain_auroc + 0.10:
        delta_note = "encoder increased domain separability versus canonical input"
    else:
        delta_note = "domain separability changed only modestly from input to embedding"
    return f"{head_note}; {rep_note}; {delta_note}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--target-csv", required=True)
    ap.add_argument("--source-name", required=True)
    ap.add_argument("--target-name", required=True)
    ap.add_argument("--mapping", default="configs/features/canonical_features.yaml")
    ap.add_argument("--allow-unverified-mapping", action="store_true")
    ap.add_argument("--encoder", choices=["mlp", "mlp_fair", "cnn_bilstm_attention"], default="mlp_fair")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lambda-d", type=float, default=0.1)
    ap.add_argument("--domain-loss-weight", type=float, default=0.5)
    ap.add_argument("--dann-lr", type=float, default=None)
    ap.add_argument("--sample-source", type=int, default=20000)
    ap.add_argument("--sample-target", type=int, default=20000)
    ap.add_argument("--probe-max-per-domain", type=int, default=10000)
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    set_global_seed(args.seed)
    effective_dann_lr = args.dann_lr if args.dann_lr is not None else (1e-3 if args.encoder == "mlp_fair" else 1e-4)

    source = _cap_source(load_csv(args.source_csv), args.source_name, args.sample_source, args.seed)
    target = _cap_target(load_csv(args.target_csv), args.sample_target, args.seed + 1)
    mapping = resolve_mapping(
        source.columns, target.columns, args.source_name, args.target_name,
        mapping_path=args.mapping, verified_only=not args.allow_unverified_mapping,
    )
    if len(mapping.features) < 5:
        raise SystemExit(f"Only {len(mapping.features)} mapped features; audit mapping first.")
    source, target = align_frames_with_mapping(source, target, mapping)
    split = prepare_clean_source_target(
        source, target, args.source_name, args.target_name,
        seed=args.seed, threshold=0.01, feature_alignment="exact",
    )

    result = train_clean_dann(
        split.X_source_train, split.y_source_train,
        split.X_source_val, split.y_source_val,
        split.X_target_unlabeled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=effective_dann_lr,
        lambda_d=args.lambda_d,
        domain_loss_weight=args.domain_loss_weight,
        class_weight_mode="balanced",
        patience=max(args.epochs, 1),  # audit requested epoch count; no early-stop ambiguity
        seed=args.seed,
        verbose=1,
        abort_on_collapse=False,
        encoder=args.encoder,
    )

    xs = _shape(split.X_source_val, args.encoder)
    xt = _shape(split.X_target_unlabeled, args.encoder)
    _, dom_s = result.model.predict(xs, verbose=0)
    _, dom_t = result.model.predict(xt, verbose=0)
    head = audit_domain_head_outputs(dom_s, dom_t)

    # Orientation-independent frozen probes: input view and learned embedding view.
    input_probe = frozen_domain_probe(
        split.X_source_val, split.X_target_unlabeled,
        seed=args.seed, max_per_domain=args.probe_max_per_domain,
    )
    embedder = _embedding_model(result.model, args.encoder)
    emb_s = embedder.predict(xs, verbose=0)
    emb_t = embedder.predict(xt, verbose=0)
    embedding_probe = frozen_domain_probe(
        emb_s, emb_t, seed=args.seed, max_per_domain=args.probe_max_per_domain,
    )

    interpretation = _interpret(head, input_probe, embedding_probe)
    payload = {
        "version": "0.4.7",
        "source": args.source_name,
        "target": args.target_name,
        "mapping_version": mapping.version,
        "mapping_verified_only": not args.allow_unverified_mapping,
        "selected_feature_count": len(split.selected_feature_names),
        "encoder": args.encoder,
        "lambda_d": args.lambda_d,
        "domain_loss_weight": args.domain_loss_weight,
        "dann_lr": effective_dann_lr,
        "epochs": args.epochs,
        "best_epoch": result.best_epoch,
        "best_source_val_macro_f1": result.best_val_macro_f1,
        "target_class_labels_used": False,
        "expected_domain_labels": {"source": 0, "target": 1},
        "domain_head": head.to_dict(),
        "input_domain_probe": input_probe.to_dict(),
        "embedding_domain_probe": embedding_probe.to_dict(),
        "embedding_minus_input_domain_auroc": float(
            embedding_probe.domain_auroc - input_probe.domain_auroc
        ),
        "interpretation": interpretation,
    }

    out_dir = Path(args.output_dir or f"results/diagnostics/dann_domain_head_{args.source_name}_to_{args.target_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "domain_head_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    mapping_dataframe(mapping).to_csv(out_dir / "feature_mapping.csv", index=False)
    pd.DataFrame(result.history).to_csv(out_dir / "training_history.csv", index=False)

    print("\nIDS v0.4.7 DANN DOMAIN-HEAD AUDIT")
    print("=" * 64)
    print(f"source={args.source_name} target={args.target_name} encoder={args.encoder} lambda_d={args.lambda_d}")
    print(f"expected domain labels: source=0 target=1 | target class labels used: NO")
    print("\nMODEL DOMAIN HEAD")
    print(f"  balanced accuracy          : {head.balanced_accuracy:.6f}")
    print(f"  reversed balanced accuracy : {head.reversed_balanced_accuracy:.6f}")
    print(f"  domain AUROC               : {head.domain_auroc:.6f}")
    print(f"  reversed domain AUROC      : {head.reversed_domain_auroc:.6f}")
    print(f"  source mean P(target)      : {head.source_mean_p_target:.6f}")
    print(f"  target mean P(target)      : {head.target_mean_p_target:.6f}")
    print(f"  target-source score gap    : {head.p_target_gap_target_minus_source:.6f}")
    print(f"  source domain accuracy     : {head.source_accuracy:.6f}")
    print(f"  target domain accuracy     : {head.target_accuracy:.6f}")
    print(f"  orientation status         : {head.orientation_status}")
    print("\nORIENTATION-INDEPENDENT FROZEN DOMAIN PROBES")
    print(f"  canonical input probe AUROC: {input_probe.domain_auroc:.6f}")
    print(f"  embedding probe AUROC      : {embedding_probe.domain_auroc:.6f}")
    print(f"  embedding - input AUROC    : {embedding_probe.domain_auroc - input_probe.domain_auroc:+.6f}")
    print("\nINTERPRETATION")
    print(" ", interpretation)
    print("\nDecision guide:")
    print("  * head reversed, probe ~0.5: orientation/optimization artifact; representation is aligned.")
    print("  * head reversed, probe high: domains remain separable; do not call this successful alignment.")
    print("  * head ~0.5, probe high: domain head is fooled but domain information remains in embeddings.")
    print("  * head ~0.5, probe ~0.5: genuine domain confusion/alignment is plausible.")
    if args.allow_unverified_mapping:
        print("\nWARNING: candidate/unverified mapping; diagnostic only, not paper evidence.")
    print("Saved diagnostics to:", out_dir.resolve())


if __name__ == "__main__":
    main()
