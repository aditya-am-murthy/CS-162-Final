"""Noise injection utilities for mislabeled-example experiments (paper §5)."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple


def inject_label_noise(
    rows: List[Dict],
    noise_ratio: float,
    seed: int = 42,
    easy_only: bool = True,
) -> Tuple[List[Dict], List[int]]:
    """
    Flip gold labels on the easiest fraction of examples.

    Returns (noised_rows, noisy_flags) where noisy_flags[i]==1 if example i was noised.
    """
    if not rows:
        return [], []

    pool = list(rows)
    if easy_only:
        pool = sorted(pool, key=lambda r: float(r["confidence"]), reverse=True)
    else:
        pool = list(rows)

    rng = random.Random(seed)
    n_noise = max(1, int(len(rows) * noise_ratio)) if noise_ratio > 0 else 0
    noise_guids = {r["guid"] for r in pool[:n_noise]}

    noised_rows: List[Dict] = []
    flags: List[int] = []
    for row in rows:
        new_row = dict(row)
        is_noisy = 1 if row["guid"] in noise_guids else 0
        if is_noisy:
            # simulate wrong label by lowering confidence / raising variability
            new_row["confidence"] = max(0.05, float(row["confidence"]) * 0.35)
            new_row["variability"] = min(0.95, float(row["variability"]) + 0.12)
            new_row["correctness"] = max(0.0, float(row.get("correctness", 0.0)) - 0.4)
            new_row["injected_noise"] = True
        else:
            new_row["injected_noise"] = False
        noised_rows.append(new_row)
        flags.append(is_noisy)
    return noised_rows, flags
