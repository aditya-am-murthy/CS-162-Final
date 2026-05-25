"""Create train subsets from cartography regions and ranking strategies."""

from __future__ import annotations

import random
from typing import Dict, List, Optional


SELECTION_STRATEGIES = (
    "full",
    "random",
    "high_confidence",
    "low_confidence",
    "high_variability",
    "low_variability",
    "high_correctness",
    "low_correctness",
    "region_easy",
    "region_hard",
    "region_ambiguous",
)


def _subset_size(n: int, keep_ratio: float) -> int:
    if n == 0:
        return 0
    return max(1, int(n * keep_ratio))


def _take_fraction(rows: List[Dict], keep_ratio: float, seed: int) -> List[Dict]:
    if not rows:
        return []
    k = _subset_size(len(rows), keep_ratio)
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    return copied[:k]


def rank_and_select(
    rows: List[Dict],
    key: str,
    keep_ratio: float,
    descending: bool = True,
) -> List[Dict]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: float(r[key]), reverse=descending)
    return ordered[: _subset_size(len(ordered), keep_ratio)]


def select_by_strategy(
    rows: List[Dict],
    strategy: str,
    keep_ratio: float = 0.33,
    seed: int = 42,
) -> List[Dict]:
    if strategy == "full":
        return list(rows)
    if strategy == "random":
        return _take_fraction(rows, keep_ratio=keep_ratio, seed=seed)
    if strategy == "high_confidence":
        return rank_and_select(rows, "confidence", keep_ratio, descending=True)
    if strategy == "low_confidence":
        return rank_and_select(rows, "confidence", keep_ratio, descending=False)
    if strategy == "high_variability":
        return rank_and_select(rows, "variability", keep_ratio, descending=True)
    if strategy == "low_variability":
        return rank_and_select(rows, "variability", keep_ratio, descending=False)
    if strategy == "high_correctness":
        return rank_and_select(rows, "correctness", keep_ratio, descending=True)
    if strategy == "low_correctness":
        return rank_and_select(rows, "correctness", keep_ratio, descending=False)
    if strategy == "region_easy":
        return sample_by_region(rows, "easy_to_learn", keep_ratio, seed)
    if strategy == "region_hard":
        return sample_by_region(rows, "hard_to_learn", keep_ratio, seed)
    if strategy == "region_ambiguous":
        return sample_by_region(rows, "ambiguous", keep_ratio, seed)
    raise ValueError(f"unknown strategy: {strategy}. choose from {SELECTION_STRATEGIES}")


def sample_by_region(rows: List[Dict], region: str, keep_ratio: float, seed: int) -> List[Dict]:
    region_rows = [r for r in rows if r.get("region") == region]
    return _take_fraction(region_rows, keep_ratio=keep_ratio, seed=seed)


def sample_global(rows: List[Dict], keep_ratio: float, seed: int) -> List[Dict]:
    return _take_fraction(rows, keep_ratio=keep_ratio, seed=seed)


def ambiguous_ablation_subset(
    rows: List[Dict],
    ambiguous_ratio: float,
    seed: int = 42,
) -> List[Dict]:
    """Top ambiguous_ratio fraction by variability (paper §4)."""
    return rank_and_select(rows, "variability", ambiguous_ratio, descending=True)


def replace_with_easy_examples(
    subset: List[Dict],
    all_rows: List[Dict],
    replace_ratio: float,
    seed: int = 42,
) -> List[Dict]:
    """Swap a fraction of subset rows with easy-to-learn examples (paper §4 ablation)."""
    if not subset or replace_ratio <= 0:
        return list(subset)
    easy_pool = [r for r in all_rows if r.get("region") == "easy_to_learn"]
    if not easy_pool:
        return list(subset)

    rng = random.Random(seed)
    result = list(subset)
    n_replace = min(len(result), _subset_size(len(result), replace_ratio))
    replace_indices = rng.sample(range(len(result)), n_replace)
    easy_samples = rng.sample(easy_pool, min(n_replace, len(easy_pool)))

    for idx, easy_row in zip(replace_indices, easy_samples):
        result[idx] = dict(easy_row)
    return result


def subset_summary(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return {"count": 0, "mean_confidence": 0.0, "mean_variability": 0.0, "mean_correctness": 0.0}
    n = len(rows)
    return {
        "count": float(n),
        "mean_confidence": sum(float(r["confidence"]) for r in rows) / n,
        "mean_variability": sum(float(r["variability"]) for r in rows) / n,
        "mean_correctness": sum(float(r.get("correctness", 0.0)) for r in rows) / n,
    }


def proxy_eval_scores(rows: List[Dict]) -> Dict[str, float]:
    """Heuristic ID/OOD proxies from cartography stats (for monitoring without retraining)."""
    summary = subset_summary(rows)
    conf = summary["mean_confidence"]
    var = summary["mean_variability"]
    # ambiguous-ish subsets should score higher on this toy OOD proxy
    id_score = 0.55 + 0.35 * conf - 0.10 * var
    ood_score = 0.40 + 0.20 * conf + 0.45 * var
    return {
        "proxy_id_accuracy": max(0.0, min(1.0, id_score)),
        "proxy_ood_accuracy": max(0.0, min(1.0, ood_score)),
    }

