from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


LABEL_CANDIDATES = ("label", "Label", "attack", "Attack", "attack_type", "Attack_type")


def find_label_column(columns) -> str:
    for candidate in LABEL_CANDIDATES:
        if candidate in columns:
            return candidate
    lower = {str(c).lower(): c for c in columns}
    for key in ("label", "attack", "attack_type"):
        if key in lower:
            return lower[key]
    raise KeyError(f"Could not find a label column. Columns include: {list(columns)[:20]}")


def attack_to_category(label: str) -> str:
    """Map CICIoT2023 fine-grained labels to the official 7 attack families + Benign.

    The mapping is intentionally string-based so it can handle common spelling and
    delimiter variants across CSV mirrors. Unknown labels are kept as 'OtherAttack'
    rather than silently folded into a known family.
    """
    raw = str(label).strip()
    s = re.sub(r"[\s_\-]+", " ", raw).lower()
    if "benign" in s or s in {"normal", "0"}:
        return "Benign"
    if "ddos" in s:
        return "DDoS"
    # Test DDoS before DoS because 'ddos' contains 'dos'.
    if re.search(r"\bdos\b", s):
        return "DoS"
    if "mirai" in s:
        return "Mirai"
    if any(x in s for x in ("recon", "scan", "host discovery", "ping sweep")):
        return "Recon"
    if any(x in s for x in ("spoof", "arp", "dns spoof")):
        return "Spoofing"
    if any(x in s for x in ("brute", "dictionary")):
        return "BruteForce"
    if any(x in s for x in ("sql", "xss", "web", "browser", "backdoor", "upload", "command injection")):
        return "WebBased"
    return "OtherAttack"


def _read_sampled_file(path: Path, sample_frac: float, seed: int) -> pd.DataFrame:
    if sample_frac <= 0 or sample_frac > 1:
        raise ValueError("sample_frac must be in (0, 1]")
    if sample_frac == 1:
        return pd.read_csv(path, low_memory=False)
    # Chunked sampling prevents a 10+ GB dataset from being loaded into RAM at once.
    chunks = []
    for i, chunk in enumerate(pd.read_csv(path, low_memory=False, chunksize=200_000)):
        n = max(1, int(round(len(chunk) * sample_frac)))
        chunks.append(chunk.sample(n=min(n, len(chunk)), random_state=seed + i))
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def prepare_ciciot2023(
    raw_dir: str | Path,
    output_csv: str | Path,
    sample_frac: float = 0.01,
    seed: int = 42,
    task: str = "binary",
    max_files: int | None = None,
) -> pd.DataFrame:
    """Create a manageable CICIoT2023 research CSV from recursively discovered files.

    Parameters
    ----------
    task:
        'binary' -> Label = 0 Benign, 1 Attack.
        'category' -> Label contains Benign plus official attack families.
        'fine' -> preserve original fine-grained labels.

    This is a research adapter, not part of the legacy conference reproduction.
    It keeps numeric features only, replaces +/-inf with NaN and drops rows with
    missing numeric values. Every run should log sample_frac and selected files.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.rglob("*.csv"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    frames: list[pd.DataFrame] = []
    for i, path in enumerate(files):
        df = _read_sampled_file(path, sample_frac=sample_frac, seed=seed + 1000 * i)
        if len(df):
            frames.append(df)
    data = pd.concat(frames, ignore_index=True, copy=False)
    label_col = find_label_column(data.columns)
    raw_labels = data[label_col].astype(str)

    if task == "binary":
        labels = raw_labels.map(lambda x: 0 if attack_to_category(x) == "Benign" else 1)
    elif task == "category":
        labels = raw_labels.map(attack_to_category)
    elif task == "fine":
        labels = raw_labels
    else:
        raise ValueError("task must be one of: binary, category, fine")

    X = data.drop(columns=[label_col]).select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    valid = ~X.isna().any(axis=1)
    out = X.loc[valid].reset_index(drop=True)
    out["Label"] = labels.loc[valid].reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
