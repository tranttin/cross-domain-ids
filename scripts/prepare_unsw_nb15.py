#!/usr/bin/env python
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from ids.data.unsw_nb15 import prepare_unsw_nb15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/unsw_nb15")
    ap.add_argument("--output-dir", default="data/processed/unsw_nb15")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    args = ap.parse_args()
    train, test = prepare_unsw_nb15(args.raw_dir, args.output_dir, task=args.task)
    print("train:", train.shape, train["Label"].value_counts().to_dict())
    print("test :", test.shape, test["Label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
