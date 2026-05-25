#!/usr/bin/env python3
"""
Experiment §2 (part 1): aggregate per-epoch logs into training dynamics.

Input JSONL rows: guid, epoch, gold_label, pred_label, prob_gold
Output: guid, gold_label, num_epochs, confidence, variability, correctness
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse

from tqdm import tqdm

from ml_cartography.core.dynamics import add_epoch_observation, build_record, summarize_record
from ml_cartography.utils.io import read_jsonl, write_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_pipeline_config


def collect_dynamics(input_path: Path) -> list[dict]:
    raw_rows = read_jsonl(input_path)
    records: dict = {}

    for row in tqdm(raw_rows, desc="collecting dynamics"):
        guid = str(row["guid"])
        gold_label = int(row["gold_label"])
        if guid not in records:
            records[guid] = build_record(guid=guid, gold_label=gold_label)
        add_epoch_observation(
            records[guid],
            prob_for_gold=float(row["prob_gold"]),
            predicted_label=int(row["pred_label"]),
        )

    return [summarize_record(rec) for rec in records.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["input_epoch_logs"])
    output_path = args.output or Path(cfg["coordinates_output"])

    init_wandb(args, job_type="collect_dynamics", config={"input": str(input_path)})

    coordinates = collect_dynamics(input_path)
    write_jsonl(output_path, coordinates)

    confidences = [float(r["confidence"]) for r in coordinates]
    variabilities = [float(r["variability"]) for r in coordinates]

    if not args.no_wandb:
        import wandb

        wandb.log(
            {
                "num_examples": len(coordinates),
                "confidence_mean": sum(confidences) / len(confidences),
                "confidence_std": (sum((c - sum(confidences) / len(confidences)) ** 2 for c in confidences) / len(confidences)) ** 0.5,
                "variability_mean": sum(variabilities) / len(variabilities),
                "variability_std": (sum((v - sum(variabilities) / len(variabilities)) ** 2 for v in variabilities) / len(variabilities)) ** 0.5,
            }
        )
        wandb.save(str(output_path))

    print(f"wrote {len(coordinates)} coordinate rows to {output_path}")
    finish_wandb()


if __name__ == "__main__":
    main()
