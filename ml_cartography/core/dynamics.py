"""Collect and summarize training dynamics per training example."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DynamicsRecord:
    guid: str
    gold_label: int
    probs_for_gold: List[float]
    predicted_labels: List[int]


def build_record(guid: str, gold_label: int) -> DynamicsRecord:
    return DynamicsRecord(
        guid=guid,
        gold_label=gold_label,
        probs_for_gold=[],
        predicted_labels=[],
    )


def add_epoch_observation(
    record: DynamicsRecord, prob_for_gold: float, predicted_label: int
) -> None:
    record.probs_for_gold.append(float(prob_for_gold))
    record.predicted_labels.append(int(predicted_label))


def summarize_record(record: DynamicsRecord) -> Dict:
    probs = [float(x) for x in record.probs_for_gold]
    preds = [int(x) for x in record.predicted_labels]
    gold = int(record.gold_label)

    if probs:
        confidence = sum(probs) / len(probs)
        sq_diff = [(p - confidence) ** 2 for p in probs]
        variability = math.sqrt(sum(sq_diff) / len(sq_diff))
    else:
        confidence = 0.0
        variability = 0.0

    if preds:
        num_correct = sum(1 for pred in preds if pred == gold)
        correctness = num_correct / len(preds)
    else:
        correctness = 0.0

    return {
        "guid": record.guid,
        "gold_label": gold,
        "num_epochs": len(record.probs_for_gold),
        "confidence": confidence,
        "variability": variability,
        "correctness": correctness,
    }

