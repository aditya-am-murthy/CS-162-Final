"""Dynamic / iterative data maps (Idea #2): snapshots, trajectories, curriculum weights."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ml_cartography.analysis.data_map import annotate_regions
from ml_cartography.core.dynamics import (
    DynamicsRecord,
    add_epoch_observation,
    build_record,
    summarize_record,
)
from ml_cartography.utils.io import read_jsonl, write_jsonl


def records_to_coordinates(
    epoch_records: List[Dict],
    max_epoch: Optional[int] = None,
) -> List[Dict]:
    """Aggregate per-epoch logs up to max_epoch into cartography coordinates."""
    by_guid: Dict[str, DynamicsRecord] = {}
    for row in epoch_records:
        e = int(row["epoch"])
        if max_epoch is not None and e > max_epoch:
            continue
        guid = row["guid"]
        if guid not in by_guid:
            by_guid[guid] = build_record(guid, int(row["gold_label"]))
        add_epoch_observation(
            by_guid[guid],
            float(row["prob_gold"]),
            int(row["pred_label"]),
        )
    out = [summarize_record(r) for r in by_guid.values()]
    return annotate_regions(out)


def save_snapshot(
    coordinates: List[Dict],
    snapshot_dir: Path,
    epoch: int,
) -> Path:
    path = snapshot_dir / f"epoch_{epoch:03d}_coordinates.jsonl"
    write_jsonl(path, coordinates)
    return path


def build_region_trajectories(snapshots: List[Tuple[int, Path]]) -> List[Dict]:
    """Merge epoch snapshots into per-guid region trajectories."""
    traj: Dict[str, Dict] = {}
    for epoch, path in sorted(snapshots, key=lambda x: x[0]):
        for row in read_jsonl(path):
            guid = row["guid"]
            if guid not in traj:
                traj[guid] = {"guid": guid, "history": []}
            traj[guid]["history"].append(
                {
                    "epoch": epoch,
                    "confidence": float(row["confidence"]),
                    "variability": float(row["variability"]),
                    "region": row.get("region", "mixed"),
                }
            )
    return list(traj.values())


def curriculum_weights_from_coordinates(
    coordinates: List[Dict],
    ambiguous_boost: float = 2.5,
    easy_scale: float = 0.4,
    hard_boost: float = 1.2,
) -> Dict[str, float]:
    """Sample weights for adaptive curriculum (upsample ambiguous, downsample easy)."""
    weights: Dict[str, float] = {}
    for row in coordinates:
        region = row.get("region", "mixed")
        w = 1.0
        if region == "ambiguous":
            w = ambiguous_boost
        elif region == "easy_to_learn":
            w = easy_scale
        elif region == "hard_to_learn":
            w = hard_boost
        weights[row["guid"]] = w
    return weights


def guid_weights_to_sample_weights(guids_in_dataset: List[str], guid_weights: Dict[str, float]) -> List[float]:
    return [guid_weights.get(g, 1.0) for g in guids_in_dataset]


def append_training_metric(path: Path, metrics: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metrics) + "\n")
