#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def version(pkg):
    try:
        from importlib.metadata import version as v
        return v(pkg)
    except Exception:
        return "NOT INSTALLED"


def main():
    print("Python executable:", sys.executable)
    print("Python prefix    :", sys.prefix)
    print("Python version   :", sys.version.replace("\n", " "))
    for pkg in ["numpy", "pandas", "scikit-learn", "tensorflow", "kaggle", "pyyaml"]:
        print(f"{pkg:17s}: {version(pkg)}")
    kaggle = shutil.which("kaggle")
    print("Kaggle CLI       :", kaggle or "NOT FOUND")
    if kaggle:
        subprocess.run([kaggle, "--version"], check=False)
    try:
        import tensorflow as tf
        tf_file = Path(tf.__file__).resolve()
        prefix = Path(sys.prefix).resolve()
        under_prefix = prefix == tf_file or prefix in tf_file.parents
        print("TensorFlow file  :", tf_file)
        print("TF in this venv  :", "YES" if under_prefix else "NO  <-- FIX ENVIRONMENT BEFORE FINAL RUNS")
        print("TensorFlow devices:", tf.config.list_physical_devices())
        print("TensorFlow GPUs   :", tf.config.list_physical_devices("GPU"))
        if not under_prefix:
            print("\nWARNING: TensorFlow is imported from another Python environment.")
            print("Check PYTHONPATH, PyCharm interpreter, and stale activated environments.")
    except Exception as exc:
        print("TensorFlow check failed:", exc)


if __name__ == "__main__":
    main()
