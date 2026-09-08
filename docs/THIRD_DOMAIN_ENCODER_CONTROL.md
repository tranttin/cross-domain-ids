# Third-domain encoder control (v0.4.5)

## Why this control exists

The recovered CNN--BiLSTM--Attention model was designed and reproduced on CIC-style
flow vectors. In the CICIoT2023 transfer path, semantic harmonization leaves a short
canonical **tabular** feature vector. Treating adjacent canonical columns as if they
formed a temporal sequence can impose an inappropriate inductive bias.

v0.4.5 therefore adds a small MLP as an **architecture control**. It is not a new
proposed IDS and it does not replace the historical backbone. The control separates
three questions:

1. Does the harmonized feature space contain transferable signal?
2. Does the sequence-style encoder preserve that signal?
3. Does DANN help or hurt once the encoder is compatible with a tabular canonical view?

## Models

- `cnn_bilstm_attention`: recovered sequence-style backbone.
- `mlp`: Dense(64) -> BN/ReLU/Dropout -> Dense(32) -> binary task head.
- `mlp + dann`: the same 32-D MLP embedding followed by a GRL/domain head.

All clean models keep the v0.4.4 safeguards:

- source-train class weighting only;
- checkpointing by source-validation Macro-F1 at a source-selected threshold;
- target adaptation labels are never used;
- optional GMM/valley target thresholding is fitted only on the unlabelled target-adaptation pool;
- target-test labels are evaluation only.

## Decision table

| MLP source-only | MLP+DANN | Interpretation |
|---|---|---|
| strong | stronger/similar | sequence encoder mismatch; DANN can help with a compatible tabular encoder |
| strong | weaker | sequence encoder mismatch plus negative transfer from vanilla DANN |
| weak | weak | revisit harmonization/preprocessing before architecture claims |
| weak | stronger | adaptation helps, but source representation/training remains weak |

Do not promote the MLP to the proposed architecture based on a diagnostic smoke run.
Final paper evidence requires a verified feature mapping, full training, repeated seeds,
and the same target-label-free protocol.
