"""Training dynamics for RLHF / DPO preference pairs (Idea #1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PreferenceDynamicsRecord:
    guid: str
    prob_chosen_scores: List[float]
    reward_margins: List[float]
    preferred_wins: List[int]


def build_preference_record(guid: str) -> PreferenceDynamicsRecord:
    return PreferenceDynamicsRecord(
        guid=guid,
        prob_chosen_scores=[],
        reward_margins=[],
        preferred_wins=[],
    )


def add_preference_epoch(
    record: PreferenceDynamicsRecord,
    prob_chosen: float,
    reward_margin: float,
    preferred_win: int,
) -> None:
    record.prob_chosen_scores.append(float(prob_chosen))
    record.reward_margins.append(float(reward_margin))
    record.preferred_wins.append(int(preferred_win))


def summarize_preference_record(record: PreferenceDynamicsRecord) -> Dict:
    probs = record.prob_chosen_scores
    margins = record.reward_margins
    wins = record.preferred_wins

    if probs:
        confidence = sum(probs) / len(probs)
        sq = [(p - confidence) ** 2 for p in probs]
        variability = math.sqrt(sum(sq) / len(sq))
    else:
        confidence = 0.0
        variability = 0.0

    if wins:
        preference_strength = sum(wins) / len(wins)
    else:
        preference_strength = 0.0

    mean_margin = sum(margins) / len(margins) if margins else 0.0

    return {
        "guid": record.guid,
        "num_epochs": len(probs),
        "confidence": confidence,
        "variability": variability,
        "preference_strength": preference_strength,
        "mean_reward_margin": mean_margin,
    }


def preference_epoch_rows_to_coordinates(epoch_rows: List[Dict]) -> List[Dict]:
    by_guid: Dict[str, PreferenceDynamicsRecord] = {}
    for row in epoch_rows:
        guid = str(row["guid"])
        if guid not in by_guid:
            by_guid[guid] = build_preference_record(guid)
        preferred = 1 if float(row.get("reward_margin", 0)) > 0 else 0
        add_preference_epoch(
            by_guid[guid],
            float(row["prob_chosen"]),
            float(row.get("reward_margin", 0)),
            preferred,
        )
    return [summarize_preference_record(r) for r in by_guid.values()]
