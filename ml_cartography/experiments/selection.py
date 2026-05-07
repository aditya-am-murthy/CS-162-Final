"""Create train subsets from cartography regions."""

from __future__ import annotations

import random
from typing import Dict, List


def _take_fraction(rows: List[Dict], keep_ratio: float, seed: int) -> List[Dict]:
    if not rows:
        return []
    k = max(1, int(len(rows) * keep_ratio))
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    return copied[:k]


def sample_by_region(rows: List[Dict], region: str, keep_ratio: float, seed: int) -> List[Dict]:
    region_rows = [r for r in rows if r.get("region") == region]
    return _take_fraction(region_rows, keep_ratio=keep_ratio, seed=seed)


def sample_global(rows: List[Dict], keep_ratio: float, seed: int) -> List[Dict]:
    return _take_fraction(rows, keep_ratio=keep_ratio, seed=seed)

