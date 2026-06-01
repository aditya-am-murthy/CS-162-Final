"""Preference data maps (Idea #1): regions for chosen/rejected RLHF-style pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def assign_preference_region(confidence: float, variability: float) -> str:
    # confidence = mean prob chosen beats rejected; variability = std across epochs
    if confidence >= 0.72 and variability <= 0.12:
        return "easy_preference"
    if confidence <= 0.38 and variability <= 0.12:
        return "hard_preference"
    if variability >= 0.16:
        return "ambiguous_preference"
    return "mixed_preference"


def annotate_preference_regions(rows: List[Dict]) -> List[Dict]:
    out = []
    for row in rows:
        tagged = dict(row)
        tagged["region"] = assign_preference_region(
            float(row["confidence"]), float(row["variability"])
        )
        out.append(tagged)
    return out


def save_preference_map_plot(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "easy_preference": "#4caf50",
        "hard_preference": "#f44336",
        "ambiguous_preference": "#2196f3",
        "mixed_preference": "#9e9e9e",
    }
    x = [float(r["variability"]) for r in rows]
    y = [float(r["confidence"]) for r in rows]
    c = [colors.get(r.get("region", "mixed_preference"), "#9e9e9e") for r in rows]

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=c, alpha=0.7, s=22)
    plt.xlabel("Variability (std of preference confidence)")
    plt.ylabel("Confidence (mean P(chosen > rejected))")
    plt.title("Preference Data Map")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
