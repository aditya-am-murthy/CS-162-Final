#!/usr/bin/env bash
# Polish existing dynamics into paper-style maps (adaptive/hard-limit/equal-thirds). One-line:
#   bash scripts/run_exp_03_polish_maps.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults

SESSION="${SESSION:-cs162-exp-polish-maps}"
RUN_DIR="${RUN_DIR:-results/20260609_074628_snli_winogrande_roberta-large}"
WANDB_GROUP="${WANDB_GROUP:-${EXP_WANDB_GROUP}}"
LOG="$(exp_log_file exp_03_polish_maps)"
ROOT="$(exp_repo_root)"
MAP_INPUT="${MAP_INPUT:-$RUN_DIR/fixed-maps/adaptive/20260609_074628_cartography_with_regions.jsonl}"

CMD="cd '$ROOT' && $PYTHON scripts/rebuild_fixed_maps.py '$RUN_DIR' && \
  $PYTHON scripts/07_generate_insight_figures.py \
    --input '$MAP_INPUT' \
    --run-id '$(basename "$RUN_DIR")' \
    --wandb-group '$WANDB_GROUP' --wandb-run-name polish_maps_winogrande && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  bash scripts/plot_from_metrics_csv.py --experiment-tag '$WANDB_GROUP' && \
  echo '=== map polish complete ==='; bash"

exp_tmux_start "$SESSION" polish-maps "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "run_dir: $RUN_DIR" \
  "map_input: $MAP_INPUT" \
  "log: $LOG"
