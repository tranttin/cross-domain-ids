# Legacy vs Clean Protocol

## Why there are two paths

The recovered notebooks contain enough detail to reproduce the historical conference
experiments, but several behaviors should not be carried into new work. Therefore the
repo treats old reproduction and new research as separate protocols.

| Item | Legacy reproduction | Clean research |
|---|---|---|
| Common feature order | Python set intersection | sorted / explicit saved order |
| Feature selector/scaler | always fit on 2018 train in cross-domain notebook | fit on source train only |
| Source validation | Keras `validation_split` / historical behavior | explicit source validation split |
| Target labels in DANN training | **yes** | **no** |
| Target labels in model selection | historical behavior can contaminate interpretation | forbidden |
| 2018->2017 evaluation | full 2017 | held-out target test |
| 2017->2018 evaluation | 2018 holdout | held-out target test |
| Scientific role | reproduce historical numbers | basis for thesis/journal claims |

## Recovered cross-domain protocol

The original notebook:

1. computes 27 common columns using a Python set;
2. splits both 2018 and 2017;
3. fits `VarianceThreshold(0.01)` on **2018 train**;
4. fits `RobustScaler(quantile_range=(5,95))` on **2018 train**;
5. transforms 2017 with those 2018-fitted transforms;
6. trains the baseline CNN-BiLSTM-Attention;
7. trains the DANN on source + target and supplies class labels for **both** domains.

The legacy runner preserves this behavior solely to test historical reproducibility.

## Clean UDA protocol

For each direction independently:

1. choose source and target;
2. determine common features deterministically;
3. split source into train/validation/test;
4. split target into unlabeled adaptation pool + held-out test;
5. fit selector/scaler on source train only;
6. train source-only baseline or DANN;
7. use source validation for early stopping;
8. evaluate once on held-out target labels.

A new PACDA-IDS implementation should follow the same contract so DANN and PACDA are
compared under the same protocol.
