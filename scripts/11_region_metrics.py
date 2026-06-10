#!/usr/bin/env python3
"""
Compute and compare per-strategy metrics from script 09 training results.

Reads data/processed/region_subsets/train_results.json and
data/processed/region_subsets/manifest.json, then computes 7 metrics:

  1. Val accuracy (ID)
  2. Gain vs random 33%
  3. OOD proxy score (from manifest)
  4. Mean variability of subset
  5. Mean correctness of subset
  6. Size efficiency ratio (val_accuracy / subset_fraction)
  7. Seed std across restarts

Writes results/region_metrics.json and prints a comparison table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from ml_cartography.utils.io import write_json
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_pipeline_config, use_wandb


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_manifest_index(manifest: dict) -> dict[str, dict]:
    """Map entry name -> manifest entry."""
    return {e["name"]: e for e in manifest.get("entries", [])}


def _compute_metrics(
    results: list[dict],
    manifest_index: dict[str, dict],
) -> dict[str, dict]:
    """Group results by strategy, compute all 7 metrics."""
    by_strategy: dict[str, list[dict]] = {}
    for r in results:
        key = r.get("strategy") or r.get("subset", "unknown")
        by_strategy.setdefault(key, []).append(r)

    metrics: dict[str, dict] = {}
    for strategy, runs in by_strategy.items():
        accs = [float(r["final_val_accuracy"]) for r in runs if "final_val_accuracy" in r]
        val_acc = _mean(accs)
        seed_std = _std(accs)

        subset_name = runs[0].get("subset", "")
        entry = manifest_index.get(subset_name, {})
        keep_ratio = float(entry.get("keep_ratio", 1.0))
        mean_var = float(entry.get("mean_variability", 0.0))
        mean_corr = float(entry.get("mean_correctness", 0.0))
        proxy_ood = float(entry.get("proxy_ood_accuracy", 0.0))

        size_efficiency = val_acc / keep_ratio if keep_ratio > 0 else 0.0

        metrics[strategy] = {
            "val_accuracy": val_acc,
            "seed_std": seed_std,
            "proxy_ood_accuracy": proxy_ood,
            "mean_variability": mean_var,
            "mean_correctness": mean_corr,
            "keep_ratio": keep_ratio,
            "size_efficiency_ratio": size_efficiency,
            "n_runs": len(runs),
        }

    # gain vs random baseline
    random_acc = metrics.get("random", {}).get("val_accuracy", 0.0)
    for strategy, m in metrics.items():
        m["gain_vs_random"] = m["val_accuracy"] - random_acc

    return metrics


def _print_table(metrics: dict[str, dict]) -> None:
    col_order = [
        "val_accuracy",
        "gain_vs_random",
        "proxy_ood_accuracy",
        "mean_variability",
        "mean_correctness",
        "size_efficiency_ratio",
        "seed_std",
    ]
    headers = [
        "strategy",
        "val_acc",
        "gain_rnd",
        "ood_proxy",
        "mean_var",
        "mean_corr",
        "size_eff",
        "seed_std",
    ]

    # sort by val_accuracy descending
    rows = sorted(metrics.items(), key=lambda kv: kv[1]["val_accuracy"], reverse=True)

    col_w = [max(len(h), 14) for h in headers]
    col_w[0] = max(len(h) for h in [r[0] for r in rows] + [headers[0]]) + 2

    def fmt(v: float) -> str:
        return f"{v:.4f}"

    sep = "  ".join("-" * w for w in col_w)
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_line)
    print(sep)
    for strategy, m in rows:
        cells = [strategy] + [fmt(m[c]) for c in col_order]
        print("  ".join(c.ljust(w) for c, w in zip(cells, col_w)))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/processed/region_subsets/train_results.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/region_subsets/manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/region_metrics.json"),
    )
    add_wandb_args(parser)
    args = parser.parse_args()

    if not args.results.exists():
        raise SystemExit(
            f"train_results.json not found at {args.results}\n"
            "Run: python scripts/09_region_finetune.py --train"
        )
    if not args.manifest.exists():
        raise SystemExit(f"manifest.json not found at {args.manifest}")

    train_data = _load_json(args.results)
    manifest = _load_json(args.manifest)

    results = train_data.get("results", train_data) if isinstance(train_data, dict) else train_data
    manifest_index = _build_manifest_index(manifest)

    metrics = _compute_metrics(results, manifest_index)

    init_wandb(
        args,
        job_type="region_metrics",
        config={"results": str(args.results), "manifest": str(args.manifest)},
    )

    _print_table(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, metrics)
    print(f"wrote {args.output}")

    if use_wandb(args):
        import wandb

        for strategy, m in metrics.items():
            wandb.log({f"{strategy}/{k}": v for k, v in m.items()})

    finish_wandb()


if __name__ == "__main__":
    main()
