#!/usr/bin/env python3
"""Run the full Dataset Cartography experiment pipeline with tqdm + W&B."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tqdm import tqdm

from scripts.common import add_wandb_args, finish_wandb, init_wandb

# (name, script, extra_args, use_config)
PIPELINE_STEPS = [
    ("00_generate_toy_epoch_logs", "scripts/00_generate_toy_epoch_logs.py", [], False),
    ("01_collect_dynamics", "scripts/01_collect_dynamics.py", [], True),
    ("02_build_data_map", "scripts/02_build_data_map.py", [], True),
    ("03_select_subsets", "scripts/03_select_subsets.py", ["--run-all-strategies"], True),
    ("04_detect_mislabeled", "scripts/04_detect_mislabeled.py", [], True),
    ("05_uncertainty_checks", "scripts/05_uncertainty_checks.py", [], True),
    ("06_ambiguous_ablation", "scripts/06_ambiguous_ablation.py", [], True),
    ("07_generate_insight_figures", "scripts/07_generate_insight_figures.py", [], False),
]


def _run_step(
    script: Path, extra_args: list[str], wandb_args: list[str], repo_root: Path
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(script), *extra_args, *wandb_args]
    subprocess.run(cmd, check=True, cwd=repo_root, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-toy-generation", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/example_pipeline.json"))
    add_wandb_args(parser)
    args = parser.parse_args()

    init_wandb(
        args,
        job_type="full_pipeline",
        config={"config": str(args.config), "steps": [s[0] for s in PIPELINE_STEPS]},
    )

    wandb_args: list[str] = []
    if args.no_wandb:
        wandb_args.append("--no-wandb")
    else:
        wandb_args.extend(["--wandb-project", args.wandb_project])
        if args.wandb_run_name:
            wandb_args.extend(["--wandb-run-name", args.wandb_run_name])
        if args.wandb_entity:
            wandb_args.extend(["--wandb-entity", args.wandb_entity])

    repo_root = Path(__file__).resolve().parents[1]
    steps = PIPELINE_STEPS[1:] if args.skip_toy_generation else PIPELINE_STEPS

    for name, rel_script, extra_args, use_config in tqdm(steps, desc="pipeline"):
        script = repo_root / rel_script
        step_args = list(extra_args)
        if use_config:
            step_args.extend(["--config", str(args.config)])
        print(f"\n=== {name} ===")
        _run_step(script, step_args, wandb_args, repo_root)

    if not args.no_wandb:
        import wandb

        wandb.log({"pipeline_status": "completed"})

    print("\nAll experiments finished.")
    finish_wandb()


if __name__ == "__main__":
    main()
