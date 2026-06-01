"""Tools for plotting and slicing cartography data maps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR = _REPO_ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_THRESHOLDS = {
    "easy_confidence_min": 0.75,
    "hard_confidence_max": 0.35,
    "low_variability_max": 0.10,
    "ambiguous_variability_min": 0.18,
}


def fit_region_thresholds(rows: List[Dict]) -> Dict[str, float]:
    """Fit region cutoffs from the current dataset instead of fixed SNLI-like values."""
    if not rows:
        return dict(DEFAULT_THRESHOLDS)

    conf = np.array([float(r["confidence"]) for r in rows], dtype=float)
    var = np.array([float(r["variability"]) for r in rows], dtype=float)

    thresholds = {
        "easy_confidence_min": float(np.quantile(conf, 0.75)),
        "hard_confidence_max": float(np.quantile(conf, 0.25)),
        "low_variability_max": float(np.quantile(var, 0.40)),
        "ambiguous_variability_min": float(np.quantile(var, 0.80)),
    }

    thresholds["easy_confidence_min"] = max(
        thresholds["easy_confidence_min"],
        thresholds["hard_confidence_max"] + 0.05,
    )
    thresholds["ambiguous_variability_min"] = max(
        thresholds["ambiguous_variability_min"],
        thresholds["low_variability_max"] + 0.02,
    )
    return thresholds


def assign_region(
    confidence: float,
    variability: float,
    thresholds: Dict[str, float] | None = None,
) -> str:
    limits = thresholds or DEFAULT_THRESHOLDS
    if (
        confidence >= limits["easy_confidence_min"]
        and variability <= limits["low_variability_max"]
    ):
        return "easy_to_learn"
    if (
        confidence <= limits["hard_confidence_max"]
        and variability <= limits["low_variability_max"]
    ):
        return "hard_to_learn"
    if variability >= limits["ambiguous_variability_min"]:
        return "ambiguous"
    return "mixed"


def annotate_regions(
    rows: List[Dict],
    thresholds: Dict[str, float] | None = None,
) -> List[Dict]:
    limits = thresholds or fit_region_thresholds(rows)
    tagged: List[Dict] = []
    for row in rows:
        new_row = dict(row)
        new_row["region"] = assign_region(
            float(row["confidence"]),
            float(row["variability"]),
            thresholds=limits,
        )
        tagged.append(new_row)
    return tagged


def summarize_regions(rows: List[Dict], thresholds: Dict[str, float]) -> Dict[str, object]:
    counts: Dict[str, int] = {
        "easy_to_learn": 0,
        "hard_to_learn": 0,
        "ambiguous": 0,
        "mixed": 0,
    }
    for row in rows:
        counts[row["region"]] = counts.get(row["region"], 0) + 1

    total = max(1, len(rows))
    shares = {key: value / total for key, value in counts.items()}
    return {
        "thresholds": thresholds,
        "counts": counts,
        "shares": shares,
        "num_examples": len(rows),
    }


def prepare_region_annotations(
    rows: List[Dict],
    thresholds: Dict[str, float] | None = None,
) -> Tuple[List[Dict], Dict[str, object]]:
    limits = thresholds or fit_region_thresholds(rows)
    tagged = annotate_regions(rows, thresholds=limits)
    summary = summarize_regions(tagged, thresholds=limits)
    return tagged, summary


def save_data_map_plot(
    rows: List[Dict],
    output_path: Path,
    thresholds: Dict[str, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limits = thresholds or fit_region_thresholds(rows)
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
    plt.scatter(x, y, c=c, alpha=0.55, s=18, linewidths=0)
    plt.axvline(
        limits["low_variability_max"],
        color="#616161",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    plt.axvline(
        limits["ambiguous_variability_min"],
        color="#1565c0",
        linestyle=":",
        linewidth=1,
        alpha=0.8,
    )
    plt.axhline(
        limits["easy_confidence_min"],
        color="#2e7d32",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    plt.axhline(
        limits["hard_confidence_max"],
        color="#c62828",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    plt.xlabel("Variability (std of gold-label probability)")
    plt.ylabel("Confidence (mean gold-label probability)")
    plt.title("Dataset Cartography Data Map")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
