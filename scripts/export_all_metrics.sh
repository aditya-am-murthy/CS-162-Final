#!/usr/bin/env bash
# One-line export: local training JSONL + wandb cache -> combined CSVs.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults

TAG="${1:-${EXP_WANDB_GROUP:-paper-reproduction}}"
ROOT="$(exp_repo_root)"

"$PYTHON" "$ROOT/scripts/export_experiment_metrics_csv.py" \
  --tag "$TAG" \
  --output "$ROOT/results/experiment_metrics_history.csv" \
  --summary-output "$ROOT/results/experiment_metrics_summary.csv"

"$PYTHON" "$ROOT/scripts/export_wandb_metrics_csv.py" \
  --group "$TAG" \
  --since all \
  --output "$ROOT/results/wandb_metrics_history.csv" \
  --summary-output "$ROOT/results/wandb_metrics_summary.csv"

echo "Local metrics:  $ROOT/results/experiment_metrics_{history,summary}.csv"
echo "W&B metrics:    $ROOT/results/wandb_metrics_{history,summary}.csv"
