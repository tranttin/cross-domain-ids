#!/usr/bin/env python3
"""Capture reproducibility metadata for the manuscript environment section."""
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PACKAGES = ["numpy", "pandas", "sklearn", "tensorflow", "xgboost", "catboost", "scipy", "yaml"]


def version_of(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as e:
        return f"unavailable: {e.__class__.__name__}"


def command(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (p.stdout or p.stderr).strip()
    except Exception as e:
        return f"unavailable: {e}"


def main():
    out = Path("results/environment")
    out.mkdir(parents=True, exist_ok=True)
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {p: version_of(p) for p in PACKAGES},
        "nvidia_smi": command(["nvidia-smi"]),
    }
    try:
        import tensorflow as tf
        info["tensorflow_build"] = tf.sysconfig.get_build_info()
        info["tensorflow_gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except Exception:
        pass
    (out / "environment_report.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    lines = [f"{k}: {v}" for k, v in info.items() if k not in {"packages", "nvidia_smi", "tensorflow_build"}]
    lines.append("packages:")
    lines += [f"  {k}: {v}" for k, v in info["packages"].items()]
    lines.append("nvidia-smi:")
    lines.append(info["nvidia_smi"])
    if "tensorflow_build" in info:
        lines.append("tensorflow build:")
        lines.append(json.dumps(info["tensorflow_build"], indent=2, default=str))
    (out / "environment_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(out / "environment_report.txt")

if __name__ == "__main__":
    main()
