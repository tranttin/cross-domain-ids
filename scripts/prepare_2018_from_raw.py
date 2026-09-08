#!/usr/bin/env python
import _bootstrap  # noqa: F401
from ids.data.cic2018 import build_cleaned_40k

if __name__ == "__main__":
    df, mapping = build_cleaned_40k("data/raw/cse_cic_ids2018", "data/processed/cleaned_data_sampled.csv")
    print(df.shape)
    print("Label mapping:", mapping)
