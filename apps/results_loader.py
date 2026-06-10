from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def list_runs(results_root: Path = RESULTS) -> list[str]:
    if not results_root.is_dir():
        return []
    run_names: list[str] = []
    for path in results_root.iterdir():
        if path.is_dir():
            run_names.append(path.name)
            continue
        if path.is_file() and path.suffix == ".json":
            run_names.append(path.name)
    return sorted(run_names, reverse=True)


def get_run_dir(run_id: str, results_root: Path = RESULTS) -> Path:
    return results_root / run_id


def run_exists(run_id: str, results_root: Path = RESULTS) -> bool:
    return get_run_dir(run_id, results_root).exists()


def get_run_kind(run_path: Path) -> str:
    if run_path.is_dir():
        return "published_run"
    if run_path.is_file() and run_path.suffix == ".json":
        return "json_report"
    return "unknown"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(run_dir: Path) -> dict[str, Any] | None:
    if run_dir.is_file():
        return _load_json(run_dir)
    return _load_json(run_dir / "manifest.json")


def load_config(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.is_dir():
        return None
    return _load_json(run_dir / "config.json")


def load_summary(run_dir: Path) -> dict[str, Any] | None:
    if not run_dir.is_dir():
        return None
    return _load_json(run_dir / "summary.json")


def list_figure_paths(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        return []
    fig_dir = run_dir / "figures"
    if not fig_dir.is_dir():
        return []
    return sorted(fig_dir.glob("*.png"))


def has_region_rows(run_dir: Path) -> bool:
    if not run_dir.is_dir():
        return False
    dynamics_path = run_dir / "dynamics" / "cartography_with_regions.jsonl"
    return dynamics_path.is_file()


def preview_region_rows(
    run_dir: Path,
    region: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    if not run_dir.is_dir():
        return []
    dynamics_path = run_dir / "dynamics" / "cartography_with_regions.jsonl"
    if not dynamics_path.is_file():
        return []

    rows: list[dict[str, Any]] = []
    with dynamics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("region") == region:
                rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def summarize_report_rows(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        return {}

    datasets = sorted({str(row.get("dataset", "unknown")) for row in rows})
    strategies = sorted({str(row.get("strategy", "unknown")) for row in rows})
    accuracies = [
        float(row["final_val_accuracy"])
        for row in rows
        if row.get("final_val_accuracy") is not None
    ]
    best_accuracy = max(accuracies) if accuracies else None

    return {
        "num_rows": len(rows),
        "datasets": datasets,
        "strategies": strategies,
        "best_accuracy": best_accuracy,
    }
