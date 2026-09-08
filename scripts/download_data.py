#!/usr/bin/env python
"""Download datasets registered in configs/datasets.yaml.

Examples
--------
python scripts/download_data.py --list
python scripts/download_data.py --dataset legacy
python scripts/download_data.py --dataset ciciot2023
python scripts/download_data.py --dataset unsw_nb15
python scripts/download_data.py --dataset cse_cic_ids2018_raw

Kaggle credentials are required for Kaggle-backed entries. Large third-domain
benchmarks are opt-in; the default `legacy` mode downloads only the two processed
files required to reproduce the original conference notebooks.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from ids.data.registry import load_dataset_registry


LEGACY = ["cse_cic_ids2018_processed", "cic_ids2017_processed"]


def find_kaggle_cli() -> str:
    scripts_dir = Path(sys.executable).resolve().parent
    for candidate in (scripts_dir / "kaggle.exe", scripts_dir / "kaggle"):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("kaggle")
    if found:
        return found
    raise RuntimeError(
        "Kaggle CLI was not found. Install it into THIS environment with:\n"
        f'  "{sys.executable}" -m pip install -U kaggle\n'
        "Then verify with: kaggle --version"
    )


def run(cmd, check=True):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=check)


def target_dir(name: str) -> Path:
    if name == "cse_cic_ids2018_raw":
        return Path("data/raw/cse_cic_ids2018")
    if name in {"cse_cic_ids2018_processed", "cic_ids2017_processed"}:
        return Path("data/processed")
    return Path("data/raw") / name


def download_one(name: str, spec: dict, kaggle_cli: str | None, dry_run: bool = False):
    info = spec.get("download", {})
    kind = info.get("kind", "manual")
    out = target_dir(name)
    print(f"\n[{name}] -> {out}")
    if spec.get("official_url"):
        print("Official source:", spec["official_url"])
    if info.get("mirror"):
        print("NOTE: automatic download uses a Kaggle mirror; verify/cite the official dataset source in publications.")

    if kind == "manual":
        print("Manual download required. See docs/DATASETS.md and the official source above.")
        return
    if kind != "kaggle":
        raise ValueError(f"Unsupported download kind: {kind}")
    slug = info["slug"]
    cmd = [kaggle_cli, "datasets", "download", "-d", slug, "-p", str(out), "--unzip"]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    out.mkdir(parents=True, exist_ok=True)
    run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=[], help="Dataset key, 'legacy', or 'all-auto'. Can repeat.")
    ap.add_argument("--list", action="store_true", help="List registered datasets.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    registry = load_dataset_registry()
    if args.list:
        for name, spec in registry.items():
            kind = spec.get("download", {}).get("kind", "manual")
            print(f"{name:28s} {kind:8s} {spec.get('role','')}")
        return

    requested = args.dataset or ["legacy"]
    names: list[str] = []
    for item in requested:
        if item == "legacy":
            names.extend(LEGACY)
        elif item == "all-auto":
            names.extend([k for k, v in registry.items() if v.get("download", {}).get("kind") == "kaggle"])
        elif item in registry:
            names.append(item)
        else:
            raise SystemExit(f"Unknown dataset '{item}'. Run --list.")
    names = list(dict.fromkeys(names))

    needs_kaggle = any(registry[n].get("download", {}).get("kind") == "kaggle" for n in names)
    kaggle_cli = find_kaggle_cli() if needs_kaggle else None
    print(f"Python executable : {sys.executable}")
    if kaggle_cli:
        print(f"Kaggle CLI        : {kaggle_cli}")
        if not args.dry_run:
            run([kaggle_cli, "--version"])

    for name in names:
        download_one(name, registry[name], kaggle_cli, dry_run=args.dry_run)

    print("\nDone. Run: python scripts/inspect_dataset.py --all")


if __name__ == "__main__":
    main()
