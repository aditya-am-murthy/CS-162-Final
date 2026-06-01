#!/usr/bin/env python3
"""
Experiment §5: noise injection + linear classifier for mislabeled-example detection.

1. Inject label noise on easiest 1% (configurable).
2. Train logistic regression on confidence/variability/correctness.
3. Evaluate on held-out split and log precision/recall/F1 to W&B.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import random

from ml_cartography.experiments.noise_detection import (
    build_feature_matrix,
    evaluate_noise_detector,
    train_noise_detector,
)
from ml_cartography.experiments.noise_injection import inject_label_noise
from ml_cartography.utils.io import read_jsonl, write_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_pipeline_config, use_wandb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--noise-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["coordinates_with_regions_output"])
    noised_output = Path("data/processed/noised_coordinates.jsonl")

    init_wandb(
        args,
        job_type="detect_mislabeled",
        config={"noise_ratio": args.noise_ratio, "seed": args.seed},
    )

    rows = read_jsonl(input_path)
    noised_rows, noisy_flags = inject_label_noise(
        rows, noise_ratio=args.noise_ratio, seed=args.seed, easy_only=True
    )
    write_jsonl(noised_output, noised_rows)

    indices = list(range(len(noised_rows)))
    random.Random(args.seed).shuffle(indices)
    split_at = int(len(indices) * (1.0 - args.test_fraction))
    train_idx = indices[:split_at]
    test_idx = indices[split_at:]

    train_rows = [noised_rows[i] for i in train_idx]
    test_rows = [noised_rows[i] for i in test_idx]
    train_flags = [noisy_flags[i] for i in train_idx]
    test_flags = [noisy_flags[i] for i in test_idx]

    model = train_noise_detector(train_rows, train_flags)
    precision, recall, f1 = evaluate_noise_detector(model, test_rows, test_flags)

    clean_preds = model.predict(build_feature_matrix(rows))
    flagged_clean = sum(int(p) == 1 for p in clean_preds)

    if use_wandb(args):
        import wandb

        wandb.log(
            {
                "noise_ratio": args.noise_ratio,
                "num_noised": sum(noisy_flags),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "clean_flagged_count": flagged_clean,
            }
        )
        wandb.save(str(noised_output))

    print(
        f"noise injection: {sum(noisy_flags)}/{len(rows)} examples | "
        f"test precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}"
    )
    print(f"noised coordinates -> {noised_output}")
    finish_wandb()


if __name__ == "__main__":
    main()
