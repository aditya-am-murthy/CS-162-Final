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


def load_manifest(run_dir: Path) -> dict[str, Any] | None:
    if run_dir.is_file():
        manifest_path = run_dir
    else:
        manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
