"""Synthetic training dynamics matching Dataset Cartography paper geometry."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple


def _correctness_from_mu_sigma(mu: float, sigma: float, rng: random.Random) -> float:
    # easy -> almost always correct; hard -> rarely; ambiguous -> mixed
    if mu >= 0.72 and sigma <= 0.08:
        base = rng.uniform(0.85, 1.0)
    elif mu <= 0.38 and sigma <= 0.08:
        base = rng.uniform(0.0, 0.35)
    else:
        base = rng.uniform(0.35, 0.85)
    return max(0.0, min(1.0, base + rng.uniform(-0.08, 0.08)))


def _sample_easy(rng: random.Random) -> Tuple[float, float]:
    mu = rng.betavariate(12, 1.5) * 0.22 + 0.76
    sigma = abs(rng.gauss(0, 0.022))
    return min(0.995, mu), min(0.095, sigma)


def _sample_hard(rng: random.Random) -> Tuple[float, float]:
    mu = rng.betavariate(1.5, 10) * 0.38 + 0.02
    sigma = abs(rng.gauss(0, 0.018))
    return max(0.01, mu), min(0.09, sigma)


def _sample_ambiguous(rng: random.Random) -> Tuple[float, float]:
    # points along the bell boundary: high sigma, mu slides down as sigma grows
    t = rng.random()
    sigma = 0.10 + t * 0.32 + rng.uniform(-0.03, 0.03)
    mu = 0.92 - 0.55 * (sigma / 0.42) ** 0.85 + rng.uniform(-0.06, 0.06)
    mu = max(0.25, min(0.88, mu))
    sigma = max(0.10, min(0.44, sigma))
    return mu, sigma


def generate_paper_like_coordinates(
    num_examples: int = 8000,
    seed: int = 42,
) -> List[Dict]:
    """
    Direct (mu, sigma) samples forming the SNLI/WinoGrande-style data map bell.

    ~68% easy, ~14% hard, ~18% ambiguous (paper: easy dominates).
    """
    rng = random.Random(seed)
    n_easy = int(num_examples * 0.68)
    n_hard = int(num_examples * 0.14)
    n_amb = num_examples - n_easy - n_hard

    rows: List[Dict] = []
    idx = 0
    for _ in range(n_easy):
        mu, sigma = _sample_easy(rng)
        rows.append(_row(idx, mu, sigma, rng, region_hint="easy_to_learn"))
        idx += 1
    for _ in range(n_hard):
        mu, sigma = _sample_hard(rng)
        rows.append(_row(idx, mu, sigma, rng, region_hint="hard_to_learn"))
        idx += 1
    for _ in range(n_amb):
        mu, sigma = _sample_ambiguous(rng)
        rows.append(_row(idx, mu, sigma, rng, region_hint="ambiguous"))
        idx += 1

    rng.shuffle(rows)
    return rows


def _row(
    idx: int,
    confidence: float,
    variability: float,
    rng: random.Random,
    region_hint: str,
) -> Dict:
    correctness = _correctness_from_mu_sigma(confidence, variability, rng)
    return {
        "guid": f"paper-{idx:06d}",
        "gold_label": rng.randint(0, 2),
        "num_epochs": 6,
        "confidence": round(confidence, 5),
        "variability": round(variability, 5),
        "correctness": round(correctness, 5),
        "region": region_hint,
    }


def apply_noise_shift(rows: List[Dict], noise_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    """Return (clean_copy, noised_rows) with paper Fig 4 style shift on flipped easy points."""
    rng = random.Random(seed)
    clean = [dict(r) for r in rows]
    easy = sorted(clean, key=lambda r: float(r["confidence"]), reverse=True)
    n_flip = max(1, int(len(clean) * noise_ratio))
    flip_guids = {r["guid"] for r in easy[:n_flip]}

    noised: List[Dict] = []
    for r in clean:
        nr = dict(r)
        if r["guid"] in flip_guids:
            nr["confidence"] = max(0.02, float(r["confidence"]) * rng.uniform(0.15, 0.35))
            nr["variability"] = min(
                0.45,
                float(r["variability"]) + rng.uniform(0.06, 0.18),
            )
            nr["correctness"] = max(0.0, float(r["correctness"]) - rng.uniform(0.5, 0.9))
            nr["region"] = "hard_to_learn"
            nr["was_noised"] = True
        else:
            nr["was_noised"] = False
        noised.append(nr)
    return clean, noised
