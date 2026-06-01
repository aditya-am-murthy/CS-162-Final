"""Timestamped experiment directories for training + Streamlit-ready artifacts."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments" / "runs"
RESULTS_ROOT = REPO_ROOT / "results"


@dataclass
class ExperimentPaths:
    run_id: str
    root: Path
    models_dir: Path
    checkpoints_dir: Path
    dynamics_dir: Path
    snapshots_dir: Path
    figures_dir: Path
    logs_dir: Path
    manifest_path: Path
    config_path: Path

    @classmethod
    def create(
        cls,
        task_slug: str,
        base: Path = EXPERIMENTS_ROOT,
        run_id: Optional[str] = None,
    ) -> "ExperimentPaths":
        ts = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}_{task_slug}"
        root = base / run_id
        paths = cls(
            run_id=run_id,
            root=root,
            models_dir=root / "models" / "final",
            checkpoints_dir=root / "models" / "checkpoints",
            dynamics_dir=root / "dynamics",
            snapshots_dir=root / "dynamics" / "snapshots",
            figures_dir=root / "figures",
            logs_dir=root / "logs",
            manifest_path=root / "manifest.json",
            config_path=root / "config.json",
        )
        for d in (
            paths.models_dir,
            paths.checkpoints_dir,
            paths.dynamics_dir,
            paths.snapshots_dir,
            paths.figures_dir,
            paths.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)
        return paths

    def write_config(self, config: Dict[str, Any]) -> None:
        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def write_manifest(self, manifest: Dict[str, Any]) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def epoch_logs_path(self) -> Path:
        return self.dynamics_dir / "epoch_predictions.jsonl"

    def coordinates_path(self) -> Path:
        return self.dynamics_dir / "cartography_coordinates.jsonl"

    def regions_path(self) -> Path:
        return self.dynamics_dir / "cartography_with_regions.jsonl"

    def trajectories_path(self) -> Path:
        return self.dynamics_dir / "region_trajectories.jsonl"

    def training_metrics_path(self) -> Path:
        return self.logs_dir / "training_metrics.jsonl"

    def publish_to_results(self, results_root: Path = RESULTS_ROOT) -> Path:
        """Copy artifacts to results/<run_id>/ (report.md stays at results root)."""
        dest = results_root / self.run_id
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for name in ("manifest.json", "config.json"):
            src = self.root / name
            if src.is_file():
                shutil.copy2(src, dest / name)
        for sub in ("dynamics", "figures", "logs"):
            src = self.root / sub
            if src.is_dir():
                shutil.copytree(src, dest / sub)
        models_final = self.models_dir
        if models_final.is_dir() and any(models_final.iterdir()):
            shutil.copytree(models_final, dest / "models" / "final")
        summary = {
            "run_id": self.run_id,
            "experiment_root": str(self.root),
            "results_dir": str(dest),
        }
        with (dest / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return dest
