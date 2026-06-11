#!/usr/bin/env bash
# Fig 5: human agreement heatmap on SNLI data map (paper §6). One-line:
#   bash scripts/run_exp_07_fig5_uncertainty.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults

SESSION="${SESSION:-cs162-exp-fig5}"
FIG5_INPUT="${FIG5_INPUT:-results/20260609_074628_snli_winogrande_roberta-large/dynamics/cartography_with_regions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/paper_plots_from_metrics}"
WANDB_NAME="${WANDB_NAME:-fig5_snli_agreement}"
WANDB_GROUP="${WANDB_GROUP:-fig5-uncertainty}"
LOG="$(exp_log_file exp_07_fig5)"
ROOT="$(exp_repo_root)"

if [[ ! -f "$FIG5_INPUT" ]]; then
  echo "missing coordinates for Fig 5: $FIG5_INPUT" >&2
  echo "run bash scripts/run_exp_01_datamap_snli.sh first, or set FIG5_INPUT=" >&2
  exit 1
fi

CMD="cd '$ROOT' && $PYTHON scripts/05_uncertainty_checks.py \
    --input '$FIG5_INPUT' \
    --wandb-run-name '${WANDB_NAME}_correlations' --wandb-group '$WANDB_GROUP' && \
  $PYTHON scripts/plot_from_metrics_csv.py \
    --fig5-input '$FIG5_INPUT' \
    --output-dir '$OUTPUT_DIR' \
    --wandb-run-name '$WANDB_NAME' --wandb-group '$WANDB_GROUP' && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  echo '=== Fig 5 complete ==='; bash"

exp_tmux_start "$SESSION" fig5 "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "input: $FIG5_INPUT" \
  "output: $OUTPUT_DIR" \
  "log: $LOG" \
  "W&B: $WANDB_NAME (group $WANDB_GROUP)"
