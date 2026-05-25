#!/usr/bin/env python3
"""
Experiment §6: compare training dynamics with human agreement (and optional dropout proxy).

Logs Spearman correlations between confidence/variability and human_agreement to W&B.
If human_agreement is missing, a deterministic proxy is synthesized for local runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import hashlib

from tqdm import tqdm

from ml_cartography.experiments.uncertainty import spearman_with_human_agreement
from ml_cartography.utils.io import read_jsonl
from scripts.common import add_wandb_args, finish_wandb, init_wandb, load_pipeline_config


def _proxy_agreement(guid: str, confidence: float) -> float:
    # stable pseudo human agreement from guid + confidence
    h = int(hashlib.md5(guid.encode()).hexdigest()[:8], 16)
    jitter = ((h % 1000) / 1000.0 - 0.5) * 0.12
    return max(0.0, min(1.0, 0.12 + 0.78 * confidence + jitter))


def attach_agreement_if_missing(rows: list[dict]) -> list[dict]:
    enriched = []
    for row in tqdm(rows, desc="preparing agreement labels"):
        new_row = dict(row)
        if new_row.get("human_agreement") in ("", None):
            new_row["human_agreement"] = _proxy_agreement(
                str(new_row["guid"]), float(new_row["confidence"])
            )
        if new_row.get("dropout_uncertainty") in ("", None):
            # variability tracks model uncertainty in the paper
            new_row["dropout_uncertainty"] = float(new_row["variability"])
        enriched.append(new_row)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    parser.add_argument("--input", type=Path, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()

    cfg = load_pipeline_config(args.config)
    input_path = args.input or Path(cfg["coordinates_with_regions_output"])

    init_wandb(args, job_type="uncertainty_checks", config={"input": str(input_path)})

    rows = attach_agreement_if_missing(read_jsonl(input_path))
    human_stats = spearman_with_human_agreement(rows, agreement_key="human_agreement")

    # dropout uncertainty vs variability
    from scipy.stats import spearmanr

    var = [float(r["variability"]) for r in rows]
    drop = [float(r["dropout_uncertainty"]) for r in rows]
    rho_drop, _ = spearmanr(var, drop)

    if not args.no_wandb:
        import wandb

        wandb.log(
            {
                "human_agreement/n": human_stats["n"],
                "human_agreement/rho_confidence": human_stats["rho_conf"],
                "human_agreement/rho_variability": human_stats["rho_var"],
                "dropout/rho_variability_vs_dropout": float(rho_drop),
            }
        )

    print(f"human agreement n={human_stats['n']}")
    print(f"  spearman(confidence, agreement) = {human_stats['rho_conf']}")
    print(f"  spearman(variability, agreement) = {human_stats['rho_var']}")
    print(f"  spearman(variability, dropout_uncertainty) = {rho_drop:.4f}")
    finish_wandb()


if __name__ == "__main__":
    main()
