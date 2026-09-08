# Research Roadmap

This file is a development roadmap, not a claim that every component is already a
validated contribution.

## Phase 0 — Historical recovery

- reproduce 2018 classical baselines;
- reproduce 2018 CNN-BiLSTM-Attention;
- reproduce historical cross-domain CNN and legacy DANN;
- save the exact feature order and reproduction deltas.

Status: framework support implemented.

## Phase 1 — Correct cross-domain baseline

- clean source/target split;
- source-only model selection;
- deterministic feature order;
- source-labelled / target-unlabelled DANN;
- report Macro-F1, AUPRC, Balanced Accuracy, FPR/FNR in addition to legacy metrics;
- multiple seeds.

Status: clean DANN runner implemented; results still need to be generated and audited.

## Phase 2 — Third-domain generalization

Priority:

1. CICIoT2023 — modern IoT domain and primary third-domain candidate;
2. UNSW-NB15 — diverse non-CIC benchmark;
3. optional IoT-23 — malware-centric IoT traffic if feature alignment is defensible.

Questions to answer before training:

- What is the binary label mapping?
- What common features are semantically equivalent across domains?
- Is the experiment UDA (target traffic visible unlabeled during training) or true unseen-domain generalization (target completely unseen)?
- Is performance dominated by class prior differences?

## Phase 3 — PACDA-IDS / stronger adaptation

Use the clean UDA data contract. Candidate components already planned for the research
line include prototype alignment, a projection head, contrastive objectives, confidence
filtering/pseudo-labeling and an explicit joint loss.

Do not implement all components in one commit. Add ablations in the order:

```text
Source only
→ Clean DANN
→ + projection / contrastive
→ + prototype alignment
→ + confidence filtering
→ full PACDA-IDS
```

Each row must share the same split and target-label policy.

## Phase 4 — Robustness

`src/ids/robustness/attacks.py` provides FGSM/PGD in normalized input space as a starting
point. Before making a security claim, add feature-valid perturbation constraints:

- immutable features;
- non-negative/count constraints;
- protocol-consistent features;
- maximum realistic perturbation by feature group.

Report clean performance and attacked performance together.

## Phase 5 — Final experimental discipline

- freeze configs before final multi-seed runs;
- aggregate mean ± standard deviation;
- preserve run_id and git commit for every paper table row;
- separate legacy historical results from clean new results in manuscripts.
