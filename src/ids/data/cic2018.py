from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

EXCLUDED_FILE = "02-20-2018.csv"
RARE_LABELS = [
    "DoS attacks-Slowloris",
    "DDOS attack-LOIC-UDP",
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
    "Label",
]


def discover_csvs(raw_dir: str | Path) -> list[Path]:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.csv"))
    return [p for p in files if p.name != EXCLUDED_FILE]


def build_cleaned_40k(raw_dir: str | Path, output_csv: str | Path, seed: int = 42):
    """Reproduce the CSE-CIC-IDS2018 dataset-building cell from the notebook.

    Exact legacy sequence:
      1. load all CSVs except 02-20-2018.csv
      2. drop Timestamp
      3. remove rare/corrupt labels
      4. sample 41,500 per retained class
      5. LabelEncoder
      6. coerce numeric, inf->NaN, drop NaN
      7. sample 40,000 per class
    """
    files = discover_csvs(raw_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")
    print(f"Found {len(files)} input CSV files")
    frames = []
    for p in files:
        print("Loading:", p)
        frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True, copy=False)
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
    if "Label" not in df.columns:
        raise KeyError("Expected a Label column")

    df = df[~df["Label"].isin(RARE_LABELS)]
    counts = df["Label"].value_counts()
    too_small = counts[counts < 41500]
    if len(too_small):
        raise ValueError(f"Classes below 41,500 before cleaning: {too_small.to_dict()}")
    df = df.groupby("Label", group_keys=False).sample(n=41500, random_state=seed)

    enc = LabelEncoder()
    df = df.copy()
    df["Label"] = enc.fit_transform(df["Label"])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    counts = df["Label"].value_counts()
    too_small = counts[counts < 40000]
    if len(too_small):
        raise ValueError(f"Classes below 40,000 after cleaning: {too_small.to_dict()}")
    df = df.groupby("Label", group_keys=False).sample(n=40000, random_state=seed)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    mapping = {int(i): str(v) for i, v in enumerate(enc.classes_)}
    return df, mapping
