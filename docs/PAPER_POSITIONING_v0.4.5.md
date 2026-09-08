# Paper positioning for the v0.4.5 MLP control

## Do not rewrite the proposed model around the MLP

The CNN--BiLSTM--Attention backbone remains the historical/proposed deep architecture
for the CIC-IDS2017/CSE-CIC-IDS2018 line. The MLP is introduced only as an
**architecture control** for the heterogeneous third-domain experiment after semantic
feature harmonization produces a short canonical tabular vector.

The purpose is causal/diagnostic: distinguish failure of the canonical representation
from mismatch between that representation and a sequence-style encoder.

## Safe paper claim before final runs

> Because the harmonized CICIoT2023 transfer view is a compact tabular feature vector
> rather than an explicitly ordered temporal sequence, we include a lightweight MLP
> control encoder. This control is not proposed as a replacement for the main
> CNN--BiLSTM--Attention backbone; it isolates the effect of encoder inductive bias from
> feature harmonization and domain adaptation.

## Result interpretation after verified full runs

- MLP >> CNN: evidence of encoder/representation mismatch under the canonical tabular view.
- MLP+DANN >= MLP: DANN remains useful once the encoder is compatible with the view.
- MLP good, MLP+DANN worse: evidence of negative transfer from vanilla DANN.
- MLP weak too: revisit harmonization; do not blame the CNN encoder alone.

Do not report smoke or unverified-mapping rows in the final manuscript.
