#!/usr/bin/env python3
"""One-time move of flat results/*.png into results/<timestamp>_baseline_synthetic/."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
BASELINE_ID = "20260525_000000_baseline_synthetic"


def main() -> None:
    dest = RESULTS / BASELINE_ID
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for png in RESULTS.glob("fig*.png"):
        shutil.move(str(png), str(dest / png.name))
        moved.append(png.name)
    index = {
        "baseline_run_id": BASELINE_ID,
        "migrated_at": datetime.now().isoformat(),
        "figures": moved,
    }
    with (RESULTS / "runs_index.json").open("w", encoding="utf-8") as f:
        import json
        json.dump(index, f, indent=2)
    print(f"moved {len(moved)} figures -> {dest}")
    print("report.md remains at results/report.md")


if __name__ == "__main__":
    main()
