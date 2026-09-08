# v0.4.8 — Staged BatchNorm-frozen DANN fairness control

## Why v0.4.8 exists

The v0.4.7 zero-strength control showed that both `MLP-Fair` source-only and `MLP-Fair-DANN (lambda=0)` fail similarly on CICIoT2023 (target AUROC near zero), while their source-validation Macro-F1 remains high. Therefore the lambda=0 control did **not** reveal a DANN-gradient bug; it revealed that the LayerNorm/extra-head fairness architecture itself does not preserve the cross-domain ranking observed with the earlier BatchNorm MLP.

The successful v0.4.5 source-only MLP used source-trained BatchNorm statistics. A DANN that updates BatchNorm on mixed source+target batches introduces a second adaptation mechanism and is not a clean adversarial comparison.

## v0.4.8 protocol

1. Source-pretrain a BatchNorm MLP using source labels only.
2. Save the exact shared encoder + task-head weights.
3. Build DANN with the identical task path.
4. Load the source-pretrained weights.
5. Freeze BatchNorm moving statistics during adaptation.
6. Continue source classification while training the domain head on source/target.
7. Compare `lambda_d=0` vs `lambda_d=0.1`.

With `lambda_d=0`, the domain head can learn but no domain gradient reaches the encoder. This is the required zero-strength adversarial continuation control.

## Decision rule

- If source-only and staged `lambda=0` preserve strong target AUROC, the control is valid.
- If staged `lambda=0.1` then degrades target AUROC, negative transfer can be attributed to adversarial alignment more credibly.
- If staged `lambda=0` still collapses, do not interpret `lambda=0.1`; investigate continuation training / preprocessing.

All unverified-mapping and smoke results remain engineering diagnostics only.
