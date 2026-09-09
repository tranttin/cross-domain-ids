from __future__ import annotations

import csv
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ids.utils.config import save_yaml


def _safe_version(package: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


@dataclass
class ExperimentTracker:
    experiment: str
    seed: int
    config: dict[str, Any] = field(default_factory=dict)
    output_root: Path = Path("results")
    protocol: str = "unspecified"
    run_id: str | None = None

    def __post_init__(self):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = self.run_id or f"{stamp}_{self.experiment}_s{self.seed}"
        self.output_root = Path(self.output_root)
        self.run_dir = self.output_root / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.output_root / "figures").mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.logger = self._build_logger()
        self.metadata = self._environment_metadata()
        self.metadata.update({
            "run_id": self.run_id,
            "experiment": self.experiment,
            "protocol": self.protocol,
            "seed": self.seed,
            "started_at": datetime.now().astimezone().isoformat(),
            "command": " ".join(sys.argv),
            "cwd": os.getcwd(),
            "status": "running",
        })
        self.write_json("metadata.json", self.metadata)
        if self.config:
            save_yaml(self.config, self.run_dir / "config.yaml")
        self.logger.info("run_id=%s protocol=%s seed=%s", self.run_id, self.protocol, self.seed)

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"ids.{self.run_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        fh = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        fh.setFormatter(formatter)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.handlers[:] = [fh, sh]
        return logger

    def _environment_metadata(self) -> dict[str, Any]:
        tf_file = None
        tf_under_prefix = None
        try:
            import tensorflow as tf
            tf_file = str(Path(tf.__file__).resolve())
            prefix = Path(sys.prefix).resolve()
            tf_under_prefix = prefix == Path(tf_file).resolve() or prefix in Path(tf_file).resolve().parents
        except Exception:
            pass
        return {
            "python": sys.version,
            "python_executable": sys.executable,
            "python_prefix": sys.prefix,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git_commit": _git_commit(),
            "tensorflow_file": tf_file,
            "tensorflow_under_current_prefix": tf_under_prefix,
            "packages": {
                "numpy": _safe_version("numpy"),
                "pandas": _safe_version("pandas"),
                "scikit-learn": _safe_version("scikit-learn"),
                "tensorflow": _safe_version("tensorflow"),
            },
        }

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)
        return path

    def save_dataframe(self, name: str, df: pd.DataFrame) -> Path:
        path = self.run_dir / name
        df.to_csv(path, index=False)
        return path

    def save_history(self, history: Any, name: str = "history.csv") -> Path | None:
        if history is None:
            return None
        data = history.history if hasattr(history, "history") else history
        if not data:
            return None
        return self.save_dataframe(name, pd.DataFrame(data))

    def save_feature_names(self, names: Iterable[str], name: str = "feature_names.json") -> Path:
        return self.write_json(name, list(names))

    def save_model_summary(self, model: Any, name: str = "model_summary.txt") -> Path:
        path = self.run_dir / name
        lines: list[str] = []
        model.summary(print_fn=lines.append)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def record_metrics(self, row: dict[str, Any], append_summary: bool = True) -> None:
        row = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "protocol": self.protocol,
            "seed": self.seed,
            **row,
        }
        metrics_path = self.run_dir / "metrics.csv"
        existing = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        out = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        out.to_csv(metrics_path, index=False)
        if append_summary:
            self._append_global_summary(row)
        self.logger.info("metrics=%s", json.dumps(row, default=_json_default))

    def _append_global_summary(self, row: dict[str, Any]) -> None:
        path = self.output_root / "summary.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        # CSV append is intentionally simple. Run one experiment process at a time
        # when sharing the same results directory.
        fieldnames = list(row.keys())
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader(); w.writerow(row)
            return

        old = pd.read_csv(path)
        merged_cols = list(dict.fromkeys([*old.columns.tolist(), *fieldnames]))
        old = old.reindex(columns=merged_cols)
        new = pd.DataFrame([row]).reindex(columns=merged_cols)
        pd.concat([old, new], ignore_index=True).to_csv(path, index=False)

    def finish(self, status: str = "completed", extra: dict[str, Any] | None = None) -> None:
        self.metadata["status"] = status
        self.metadata["finished_at"] = datetime.now().astimezone().isoformat()
        self.metadata["duration_seconds"] = round(time.time() - self.start_time, 3)
        if extra:
            self.metadata.update(extra)
        self.write_json("metadata.json", self.metadata)
        self.logger.info("finished status=%s duration=%.1fs", status, self.metadata["duration_seconds"])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.finish("completed")
            return False
        self.logger.exception("experiment failed", exc_info=(exc_type, exc, tb))
        self.finish("failed", {"error": repr(exc)})
        return False
