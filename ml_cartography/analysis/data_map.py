"""Tools for plotting and slicing cartography data maps."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def assign_region(confidence: float, variability: float) -> str:
    if confidence >= 0.75 and variability <= 0.10:
        return "easy_to_learn"
    if confidence <= 0.35 and variability <= 0.10:
        return "hard_to_learn"
    if variability >= 0.18:
        return "ambiguous"
    return "mixed"


def annotate_regions(rows: List[Dict]) -> List[Dict]:
    tagged: List[Dict] = []
    for row in rows:
        new_row = dict(row)
        new_row["region"] = assign_region(
            float(row["confidence"]), float(row["variability"])
        )
        tagged.append(new_row)
    return tagged


def save_data_map_plot(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = [float(r["variability"]) for r in rows]
    y = [float(r["confidence"]) for r in rows]
    color_by_region = {
        "easy_to_learn": "#4caf50",
        "hard_to_learn": "#f44336",
        "ambiguous": "#2196f3",
        "mixed": "#9e9e9e",
    }
    c = [color_by_region.get(r.get("region", "mixed"), "#9e9e9e") for r in rows]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=c, alpha=0.7, s=22)
    plt.xlabel("Variability (std of gold-label probability)")
    plt.ylabel("Confidence (mean gold-label probability)")
    plt.title("Dataset Cartography Data Map")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

