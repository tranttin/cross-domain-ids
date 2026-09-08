# DANN domain-head audit — v0.4.6

## Why this exists

A DANN domain-head accuracy below 0.5 is ambiguous. It can mean the head is temporarily
or systematically predicting the opposite orientation, while the learned embedding may
still contain strong domain information. Conversely, a head close to 0.5 does not prove
that the embedding is domain invariant: a stronger post-hoc classifier may still recover
the domain.

## Fixed clean-UDA semantics

- source domain label = `0`
- target domain label = `1`
- target attack/benign labels are **not used** by this audit

## Two-level audit

### 1. Existing DANN domain head

Report normal and reversed balanced accuracy/AUROC plus mean `P(target)` on each domain.
A reversed AUROC substantially above the normal AUROC indicates orientation reversal at
the audited checkpoint, not automatically a data-label bug.

### 2. Frozen representation probe

Freeze the encoder, extract source-validation and target-unlabelled embeddings, and fit
a fresh balanced Logistic Regression to predict domain membership. Since the probe owns
its own orientation, it answers the actual scientific question: *does the representation
still expose domain information?*

Run the same probe on the canonical input. Comparing input-probe vs embedding-probe
AUROC shows whether DANN reduced or increased domain separability.

## Decision guide

| Model head | Frozen embedding probe | Interpretation |
|---|---|---|
| reversed | ~0.5 | head/optimization orientation artifact; alignment plausible |
| reversed | high | domains still separable; negative-transfer interpretation remains plausible |
| ~0.5 | high | head is fooled, embedding is not domain invariant |
| ~0.5 | ~0.5 | genuine alignment/domain confusion is plausible |

Do not choose checkpoints, thresholds, or paper models from target attack/benign labels.
