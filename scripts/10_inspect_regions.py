#!/usr/bin/env python3
"""
Print representative examples per cartography region for the paper write-up.

Reads data/processed/cartography_with_regions.jsonl and optionally the raw
epoch predictions JSONL for per-epoch probability trajectories.

Selection:
  easy_to_learn — top N by confidence descending
  hard_to_learn — bottom N by confidence ascending (low variability)
  ambiguous     — top N by variability descending
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.utils.io import read_jsonl
from scripts.common import load_pipeline_config


REGIONS = ("easy_to_learn", "hard_to_learn", "ambiguous")


def _sort_key(region: str):
    if region == "easy_to_learn":
        return lambda r: -float(r["confidence"])
    if region == "hard_to_learn":
        return lambda r: float(r["confidence"])
    return lambda r: -float(r["variability"])


def _load_epoch_probs(raw_logs: list[Path]) -> dict[str, list[float]]:
    """Map guid -> list of prob_gold values, one per epoch log file."""
    guid_probs: dict[str, list[float]] = {}
    for log_path in sorted(raw_logs):
        rows = read_jsonl(log_path)
        for row in rows:
            guid = str(row.get("guid", row.get("pairID", "")))
            prob = float(row.get("prob_gold", row.get("prob_true", 0.0)))
            guid_probs.setdefault(guid, []).append(prob)
    return guid_probs


def _fmt_trajectory(probs: list[float]) -> str:
    return "[" + ", ".join(f"{p:.2f}" for p in probs) + "]"


def _print_example(row: dict, epoch_probs: dict[str, list[float]]) -> None:
    guid = str(row.get("guid", row.get("pairID", "?")))
    region = row.get("region", "?")
    print(f"[{region}] guid={guid}")

    for field in ("premise", "sentence1", "sentence"):
        if field in row:
            print(f"  premise:    {row[field]!r}")
            break

    for field in ("hypothesis", "sentence2", "option1"):
        if field in row:
            print(f"  hypothesis: {row[field]!r}")
            break

    label = row.get("gold_label", row.get("label", row.get("answer", "?")))
    print(f"  gold_label: {label}")
    print(f"  confidence: {float(row.get('confidence', 0)):.4f}  "
          f"variability: {float(row.get('variability', 0)):.4f}  "
          f"correctness: {float(row.get('correctness', 0)):.4f}")

    if guid in epoch_probs:
        print(f"  prob_gold by epoch: {_fmt_trajectory(epoch_probs[guid])}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--raw-logs",
        type=Path,
        nargs="*",
        default=[],
        metavar="JSONL",
        help="Per-epoch prediction JSONL files (e.g. data/raw/epoch_predictions_*.jsonl).",
    )
    parser.add_argument("--n", type=int, default=3, help="Examples per region.")
    parser.add_argument(
        "--region",
        choices=list(REGIONS) + ["all"],
        default="all",
        help="Which region(s) to print.",
    )
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["coordinates_with_regions_output"])
    rows = read_jsonl(input_path)
    if not rows:
        raise SystemExit(f"no rows found in {input_path}")

    epoch_probs: dict[str, list[float]] = {}
    if args.raw_logs:
        epoch_probs = _load_epoch_probs(args.raw_logs)

    regions_to_show = REGIONS if args.region == "all" else (args.region,)

    for region in regions_to_show:
        region_rows = [r for r in rows if r.get("region") == region]
        if not region_rows:
            print(f"[{region}] — no examples found\n")
            continue
        top = sorted(region_rows, key=_sort_key(region))[: args.n]
        for row in top:
            _print_example(row, epoch_probs)


if __name__ == "__main__":
    main()
