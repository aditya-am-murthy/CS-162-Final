"""Bilateral 1% label-flip experiment (extra #4): easy vs hard injection arms."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ml_cartography.analysis.data_map import assign_region
from ml_cartography.experiments.selection import select_by_strategy

REGIONS = ("easy_to_learn", "hard_to_learn", "ambiguous", "mixed")


def num_labels(dataset: str) -> int:
    if dataset in ("qnli", "winogrande"):
        return 2
    if dataset in ("snli", "mnli"):
        return 3
    raise ValueError(f"unsupported dataset: {dataset}")


def flip_label(label: int, dataset: str, rng: random.Random) -> int:
    n = num_labels(dataset)
    if n == 2:
        return 1 - int(label)
    choices = [i for i in range(n) if i != int(label)]
    return rng.choice(choices)


def easy_candidates(rows: List[dict]) -> List[dict]:
    region_easy = [r for r in rows if r.get("region") == "easy_to_learn"]
    pool = region_easy or list(rows)
    return sorted(
        pool,
        key=lambda r: (
            -float(r.get("confidence", 0.0)),
            float(r.get("variability", 0.0)),
            -float(r.get("correctness", 0.0)),
        ),
    )


def hard_candidates(rows: List[dict]) -> List[dict]:
    region_hard = [r for r in rows if r.get("region") == "hard_to_learn"]
    if len(region_hard) >= max(10, int(0.005 * len(rows))):
        pool = region_hard
        return sorted(
            pool,
            key=lambda r: (
                float(r.get("confidence", 0.0)),
                float(r.get("variability", 0.0)),
                float(r.get("correctness", 0.0)),
            ),
        )
    return select_by_strategy(rows, "low_confidence", keep_ratio=1.0)


def select_flip_arm(
    rows: List[dict],
    *,
    arm: str,
    flip_ratio: float,
    dataset: str,
    seed: int,
) -> Tuple[List[dict], Dict[str, int]]:
    if arm not in ("easy", "hard"):
        raise ValueError("arm must be 'easy' or 'hard'")
    if not rows:
        raise ValueError("empty coordinate input")

    rng = random.Random(seed)
    candidates = easy_candidates(rows) if arm == "easy" else hard_candidates(rows)
    n_flip = max(1, int(len(rows) * flip_ratio)) if flip_ratio > 0 else 0
    n_flip = min(n_flip, len(candidates))
    selected = list(candidates[:n_flip])
    rng.shuffle(selected)

    flips: List[dict] = []
    overrides: Dict[str, int] = {}
    for row in selected:
        if "gold_label" not in row:
            raise ValueError("rows must include gold_label")
        guid = str(row["guid"])
        original = int(row["gold_label"])
        flipped = flip_label(original, dataset, rng)
        overrides[guid] = flipped
        flips.append(
            {
                "guid": guid,
                "arm": arm,
                "original_label": original,
                "new_label": flipped,
                "confidence": float(row.get("confidence", 0.0)),
                "variability": float(row.get("variability", 0.0)),
                "correctness": float(row.get("correctness", 0.0)),
                "region": row.get("region", "unknown"),
            }
        )
    return flips, overrides


def _region_of(row: dict) -> str:
    if row.get("region"):
        return str(row["region"])
    return assign_region(float(row["confidence"]), float(row["variability"]))


def shift_rows_for_arm(
    original_rows: List[dict],
    retrained_rows: List[dict],
    flipped_guids: Set[str],
    *,
    arm: str,
) -> List[dict]:
    original_by_guid = {str(r["guid"]): r for r in original_rows}
    out: List[dict] = []
    for row in retrained_rows:
        guid = str(row["guid"])
        if guid not in flipped_guids:
            continue
        before = original_by_guid.get(guid)
        if before is None:
            continue
        region_before = _region_of(before)
        region_after = _region_of(row)
        conf_before = float(before["confidence"])
        conf_after = float(row["confidence"])
        out.append(
            {
                "guid": guid,
                "arm": arm,
                "region_before": region_before,
                "region_after": region_after,
                "confidence_before": conf_before,
                "confidence_after": conf_after,
                "variability_before": float(before["variability"]),
                "variability_after": float(row["variability"]),
                "correctness_before": float(before.get("correctness", 0.0)),
                "correctness_after": float(row.get("correctness", 0.0)),
                "confidence_delta": conf_after - conf_before,
                "variability_delta": float(row["variability"]) - float(before["variability"]),
                "recovered": conf_after > conf_before + 0.05,
                "degraded": conf_after < conf_before - 0.05,
            }
        )
    return out


def region_transition_matrix(shift_rows: List[dict]) -> np.ndarray:
    idx = {r: i for i, r in enumerate(REGIONS)}
    mat = np.zeros((len(REGIONS), len(REGIONS)), dtype=int)
    for row in shift_rows:
        i = idx.get(row["region_before"], idx["mixed"])
        j = idx.get(row["region_after"], idx["mixed"])
        mat[i, j] += 1
    return mat


def summarize_arm_shift(shift_rows: List[dict], *, arm: str) -> Dict[str, float]:
    if not shift_rows:
        return {"arm": arm, "count": 0}

    def mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in shift_rows]))

    return {
        "arm": arm,
        "count": len(shift_rows),
        "confidence_before_mean": mean("confidence_before"),
        "confidence_after_mean": mean("confidence_after"),
        "confidence_delta_mean": mean("confidence_delta"),
        "variability_before_mean": mean("variability_before"),
        "variability_after_mean": mean("variability_after"),
        "variability_delta_mean": mean("variability_delta"),
        "pct_recovered": sum(1 for r in shift_rows if r["recovered"]) / len(shift_rows),
        "pct_degraded": sum(1 for r in shift_rows if r["degraded"]) / len(shift_rows),
        "pct_easyward": sum(
            1
            for r in shift_rows
            if REGIONS.index(r["region_after"]) > REGIONS.index(r["region_before"])
        )
        / len(shift_rows),
    }


def train_confidence_detector(
    retrained_rows: List[dict],
    flipped_guids: Set[str],
    *,
    seed: int,
    test_fraction: float = 0.3,
) -> Tuple[dict, object]:
    """Confidence-only threshold detector (paper §5). Returns (summary, predict_fn)."""
    rng = random.Random(seed)
    noisy = [r for r in retrained_rows if str(r["guid"]) in flipped_guids]
    clean = [r for r in retrained_rows if str(r["guid"]) not in flipped_guids]
    if not noisy:
        raise ValueError("no flipped rows found in retrained coordinates (guid mismatch?)")
    if len(clean) < len(noisy):
        clean = clean[: max(len(noisy), 1)]
        if len(clean) < len(noisy):
            raise ValueError("insufficient clean rows for detector training")

    rng.shuffle(noisy)
    rng.shuffle(clean)
    clean = clean[: len(noisy)]

    n_test = max(1, int(len(noisy) * test_fraction)) if len(noisy) > 1 else 0
    n_test = min(n_test, len(noisy) - 1) if len(noisy) > 1 else 0

    test_rows = noisy[:n_test] + clean[:n_test]
    test_y = [1] * n_test + [0] * n_test
    train_rows = noisy[n_test:] + clean[n_test:]
    train_y = [1] * (len(noisy) - n_test) + [0] * (len(clean) - n_test)

    train_conf = [float(r["confidence"]) for r in train_rows]

    def prf(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    candidates = sorted(set(train_conf))
    if len(candidates) > 40:
        candidates = list(np.quantile(train_conf, np.linspace(0.05, 0.95, 30)))

    best = {"threshold": candidates[0], "direction": "le", "f1": -1.0}
    for threshold in candidates:
        for direction in ("le", "ge"):
            pred = [
                int(c <= threshold) if direction == "le" else int(c >= threshold)
                for c in train_conf
            ]
            _p, _r, f1 = prf(train_y, pred)
            if f1 > best["f1"]:
                best = {"threshold": float(threshold), "direction": direction, "f1": f1}

    def predict(confidence: float) -> int:
        threshold = float(best["threshold"])
        if best["direction"] == "le":
            return int(confidence <= threshold)
        return int(confidence >= threshold)

    test_conf = [float(r["confidence"]) for r in test_rows]
    if test_rows:
        y_pred = [predict(c) for c in test_conf]
        precision, recall, f1 = prf(test_y, y_pred)
    else:
        precision = recall = f1 = 0.0

    summary = {
        "threshold": best["threshold"],
        "threshold_direction": best["direction"],
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "balanced_train_rows": len(train_rows),
        "balanced_test_rows": len(test_rows),
    }
    return summary, predict


def detector_cross_eval(
    original_rows: List[dict],
    easy_detector_predict,
    hard_detector_predict,
    easy_flipped: Set[str],
    hard_flipped: Set[str],
) -> Dict[str, object]:
    """Cross-evaluate easy-trained vs hard-trained detectors on labeled cohorts."""

    def cohort_metrics(rows: List[dict], predict_fn, label: str) -> Dict[str, float]:
        if not rows:
            return {"cohort": label, "count": 0}
        preds = [predict_fn(float(r["confidence"])) for r in rows]
        return {
            "cohort": label,
            "count": len(rows),
            "pct_predicted_noisy": sum(preds) / len(preds),
            "mean_confidence": float(np.mean([float(r["confidence"]) for r in rows])),
        }

    easy_inj = [r for r in original_rows if str(r["guid"]) in easy_flipped]
    hard_inj = [r for r in original_rows if str(r["guid"]) in hard_flipped]
    natural_hard = sorted(original_rows, key=lambda r: float(r["confidence"]))[
        : max(len(hard_inj), 1)
    ]
    clean = [
        r
        for r in original_rows
        if str(r["guid"]) not in easy_flipped and str(r["guid"]) not in hard_flipped
    ]
    rng = random.Random(42)
    rng.shuffle(clean)
    clean_sample = clean[: max(len(easy_inj), 1)]

    matrix: Dict[str, Dict[str, float]] = {}
    for det_name, predict_fn in (("easy_detector", easy_detector_predict), ("hard_detector", hard_detector_predict)):
        matrix[det_name] = {
            "on_easy_injected_original": cohort_metrics(easy_inj, predict_fn, "easy_injected")[
                "pct_predicted_noisy"
            ],
            "on_hard_injected_original": cohort_metrics(hard_inj, predict_fn, "hard_injected")[
                "pct_predicted_noisy"
            ],
            "on_natural_hard_original": cohort_metrics(natural_hard, predict_fn, "natural_hard")[
                "pct_predicted_noisy"
            ],
            "on_clean_original": cohort_metrics(clean_sample, predict_fn, "clean")["pct_predicted_noisy"],
        }

    return {
        "easy_arm_detector": matrix["easy_detector"],
        "hard_arm_detector": matrix["hard_detector"],
        "cohort_sizes": {
            "easy_injected": len(easy_inj),
            "hard_injected": len(hard_inj),
            "natural_hard": len(natural_hard),
            "clean": len(clean_sample),
        },
    }


def bilateral_comparison(
    easy_summary: Dict[str, float],
    hard_summary: Dict[str, float],
) -> Dict[str, float]:
    return {
        "confidence_delta_easy_arm": easy_summary.get("confidence_delta_mean", 0.0),
        "confidence_delta_hard_arm": hard_summary.get("confidence_delta_mean", 0.0),
        "antisymmetric_gap": (
            hard_summary.get("confidence_delta_mean", 0.0)
            - easy_summary.get("confidence_delta_mean", 0.0)
        ),
        "pct_recovered_easy_arm": easy_summary.get("pct_recovered", 0.0),
        "pct_recovered_hard_arm": hard_summary.get("pct_recovered", 0.0),
        "pct_easyward_easy_arm": easy_summary.get("pct_easyward", 0.0),
        "pct_easyward_hard_arm": hard_summary.get("pct_easyward", 0.0),
        "supports_mislabel_hypothesis": (
            hard_summary.get("confidence_delta_mean", 0.0) > 0.05
            and easy_summary.get("confidence_delta_mean", 0.0) < -0.05
        ),
    }
