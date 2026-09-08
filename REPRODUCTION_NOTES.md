# Reproduction audit: notebooks vs paper

This document records facts recovered from the original notebooks and one successful
local rerun. It is deliberately separate from the clean research protocol.

## Recovered dataset facts

- Processed CSE-CIC-IDS2018: `(360000, 79)`, 9 classes x 40,000.
- Processed CIC-IDS2017: `(137992, 78)`.
- Cross-domain binary counts in the corrected notebook cells:
  - 2017: benign 30,000; attack 107,992.
  - 2018: benign 40,000; attack 320,000.
- Common columns printed by the notebook: 27.
- `VarianceThreshold(0.01)` leaves 23 cross-domain features in the saved run.

## Legacy in-domain 2018

- train/test: 80/20, stratified, random_state=42;
- `VarianceThreshold(0.01)` fit on train;
- `RobustScaler(quantile_range=(5,95))` fit on train;
- classical models add `SelectKBest(f_classif, k=30)`;
- CNN-BiLSTM-Attention uses LR `1e-4`, batch 256, up to 50 epochs;
- the notebook uses the test split as Keras `validation_data`.

A local rerun of the recovered hybrid code produced:

```text
accuracy  = 0.8672083333
precision = 0.8740269526
recall    = 0.8672083333
f1        = 0.8647433463
```

Historical paper reference: Accuracy 0.8667, Precision 0.8738, Recall 0.8667,
F1 0.8641. This is considered a successful reproduction of the in-domain hybrid.

## Legacy cross-domain notebook

The exact recovered behavior is important:

- common features are constructed using a Python set intersection;
- both datasets are split, but selector/scaler are **always fit on 2018 train**;
- 2018->2017 baseline evaluates on full 2017;
- 2017->2018 baseline evaluates on the 2018 holdout;
- DANN uses LR `1e-4`, GRL lambda 1.0, domain loss weight 0.5;
- DANN concatenates source and target **class labels** for `cls_output` training.

The original notebook output contains these cross-domain rows:

```text
CNN 2018->2017  Accuracy 0.777733  F1 0.865768
CNN 2017->2018  Accuracy 0.597806  F1 0.720623
DANN 2018->2017 Accuracy 0.817584  F1 0.894180
DANN 2017->2018 Accuracy 0.887736  F1 0.940270
```

The DANN rows reproduce a historical computation, but they should not be described as
source-labelled/target-unlabelled UDA because target class labels enter training.

## Feature-order warning

The notebook expression `list((cols17 & cols18) - {"Label", "Label_bin"})` is hash-order
dependent. The old notebook printed the first ten common features as:

```text
Fwd_IAT_Std
Flow_IAT_Std
Bwd_PSH_Flags
Flow_IAT_Max
Bwd_IAT_Min
Bwd_IAT_Mean
CWE_Flag_Count
Idle_Std
Flow_Duration
Active_Min
```

The full historical order was not printed. Therefore the new legacy runner saves every
run's exact common-feature order and allows replaying it from a text file.

## Policy

- `legacy_*` scripts exist only to reproduce historical behavior.
- `clean_*` scripts are the basis for new thesis/journal experiments.
- Never improve a legacy score by silently changing its protocol.
