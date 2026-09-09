from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _find_split_file(root: Path, kind: str) -> Path:
    candidates = sorted(root.rglob("*.csv"))
    tokens = {
        "train": ("training-set", "training_set", "train"),
        "test": ("testing-set", "testing_set", "test"),
    }[kind]
    matches = [p for p in candidates if any(t in p.name.lower() for t in tokens)]
    if not matches:
        raise FileNotFoundError(f"Could not find UNSW-NB15 {kind} CSV under {root}")
    return matches[0]


def prepare_unsw_nb15(
    raw_dir: str | Path,
    output_dir: str | Path,
    task: str = "binary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize the author-provided UNSW-NB15 train/test CSVs.

    Preserves the provided split. Numeric features are retained; common identifier
    columns are dropped when present. For binary task the official `label` column
    is used. For multiclass task `attack_cat` is used.
    """
    raw_dir = Path(raw_dir)
    train_path = _find_split_file(raw_dir, "train")
    test_path = _find_split_file(raw_dir, "test")
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)

    label_col = "label" if task == "binary" else "attack_cat"
    if label_col not in train.columns or label_col not in test.columns:
        raise KeyError(f"Expected '{label_col}' in UNSW-NB15 files")

    drop_cols = {"id", "label", "attack_cat"}
    common = [c for c in train.columns if c in test.columns and c not in drop_cols]
    numeric = [c for c in common if pd.api.types.is_numeric_dtype(train[c]) and pd.api.types.is_numeric_dtype(test[c])]

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        X = df[numeric].replace([np.inf, -np.inf], np.nan)
        valid = ~X.isna().any(axis=1)
        out = X.loc[valid].reset_index(drop=True)
        y = df.loc[valid, label_col].reset_index(drop=True)
        if task == "binary":
            y = y.astype(int)
        else:
            y = y.astype(str).str.strip().replace({"": "Normal"})
        out["Label"] = y
        return out

    train_out, test_out = clean(train), clean(test)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(output_dir / "unsw_nb15_train.csv", index=False)
    test_out.to_csv(output_dir / "unsw_nb15_test.csv", index=False)
    return train_out, test_out
