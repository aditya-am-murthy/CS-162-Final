#!/usr/bin/env python3
"""Generate synthetic per-epoch prediction logs for local pipeline testing."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import json
import random
from tqdm import tqdm

from scripts.common import add_wandb_args, finish_wandb, init_wandb, use_wandb


def _profile_for_kind(kind: str, rng: random.Random) -> tuple[float, float]:
    if kind == "easy":
        base = rng.uniform(0.82, 0.98)
        drift = rng.uniform(-0.03, 0.03)
        return base, drift
    if kind == "hard":
        base = rng.uniform(0.08, 0.32)
        drift = rng.uniform(-0.02, 0.02)
        return base, drift
    # ambiguous
    base = rng.uniform(0.45, 0.65)
    drift = rng.uniform(-0.18, 0.18)
    return base, drift


def generate_logs(
    num_examples: int,
    num_epochs: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    kinds = (["easy"] * 55 + ["hard"] * 20 + ["ambiguous"] * 25)
    rows: list[dict] = []

    for idx in tqdm(range(num_examples), desc="generating examples"):
        guid = f"ex-{idx:05d}"
        gold_label = rng.randint(0, 2)
        kind = kinds[idx % len(kinds)]
        base, drift = _profile_for_kind(kind, rng)

        for epoch in range(1, num_epochs + 1):
            prob = base + drift * (epoch - 1) + rng.uniform(-0.04, 0.04)
            prob = max(0.01, min(0.99, prob))
            pred_label = gold_label if prob >= 0.5 else (gold_label + 1) % 3
            rows.append(
                {
                    "guid": guid,
                    "epoch": epoch,
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "prob_gold": round(prob, 5),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/epoch_predictions_toy.jsonl"),
    )
    parser.add_argument("--num-examples", type=int, default=300)
    parser.add_argument("--num-epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    add_wandb_args(parser)
    args = parser.parse_args()

    init_wandb(
        args,
        job_type="generate_toy_logs",
        config={
            "num_examples": args.num_examples,
            "num_epochs": args.num_epochs,
            "seed": args.seed,
        },
    )

    rows = generate_logs(args.num_examples, args.num_epochs, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="writing jsonl"):
            f.write(json.dumps(row) + "\n")

    if use_wandb(args):
        import wandb

        wandb.log(
            {
                "num_rows": len(rows),
                "num_examples": args.num_examples,
                "num_epochs": args.num_epochs,
            }
        )
        wandb.save(str(args.output))

    print(f"wrote {len(rows)} rows to {args.output}")
    finish_wandb()


if __name__ == "__main__":
    main()
