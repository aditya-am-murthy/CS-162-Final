#!/usr/bin/env bash
# Table 2: WinoGrande 33% subset selection (paper §3). One-line:
#   bash scripts/run_exp_04_table2_winogrande.sh
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "$0")" && pwd)/lib/experiment_env.sh"
exp_activate_conda
exp_apply_paper_defaults
[[ -f "$(dirname "$0")/lib/fast_5hour.env" ]] && source "$(dirname "$0")/lib/fast_5hour.env"
exp_require_input

if [[ "${SKIP_TRAIN_TABLE2:-0}" == "1" ]]; then
  echo "SKIP_TRAIN_TABLE2=1 — using existing results in ${TABLE2_OUTPUT:-data/processed/region_finetune_winogrande}"
  ROOT="$(exp_repo_root)"
  OUT="${TABLE2_OUTPUT:-data/processed/region_finetune_winogrande}"
  "$PYTHON" "$ROOT/scripts/11_region_metrics.py" \
    --results "$OUT/train_results.json" --manifest "$OUT/manifest.json" \
    --output "$ROOT/results/region_metrics_table2.json" --no-wandb
  bash "$ROOT/scripts/export_all_metrics.sh" "${WANDB_GROUP:-deadline-5h}"
  "$PYTHON" "$ROOT/scripts/plot_from_metrics_csv.py" --experiment-tag table2 --no-wandb
  echo "=== Table 2 complete (existing runs) ==="
  exit 0
fi

SESSION="${SESSION:-cs162-exp-table2}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed/region_finetune_winogrande}"
EPOCHS="${EPOCHS:-${EXP_EPOCHS_WINO}}"
BATCH_SIZE="${BATCH_SIZE:-${EXP_BATCH_SIZE_WINO}}"
RESTARTS="${RESTARTS:-${EXP_RESTARTS}}"
GPUS="${GPUS:-${EXP_GPUS}}"
WANDB_NAME="${WANDB_NAME:-table2_winogrande_33pct}"
WANDB_GROUP="${WANDB_GROUP:-table2-winogrande}"
LOG="$(exp_log_file exp_04_table2)"
ROOT="$(exp_repo_root)"

CMD="cd '$ROOT' && $PYTHON scripts/09_region_finetune.py \
    --train --dataset winogrande --preset '$EXP_PRESET' \
    --epochs '$EPOCHS' --batch-size '$BATCH_SIZE' \
    --learning-rate '$EXP_LEARNING_RATE' \
    --max-train-samples '$EXP_MAX_TRAIN_SAMPLES' \
    --max-eval-samples '$EXP_MAX_EVAL_SAMPLES' \
    --winogrande-config '$EXP_WINOGRANDE_CONFIG' \
    --restarts '$RESTARTS' --gpus '$GPUS' \
    --input '$EXP_INPUT' --output-dir '$OUTPUT_DIR' \
    --wandb-run-name '$WANDB_NAME' --wandb-group '$WANDB_GROUP' && \
  $PYTHON scripts/11_region_metrics.py \
    --results '$OUTPUT_DIR/train_results.json' \
    --manifest '$OUTPUT_DIR/manifest.json' \
    --output results/region_metrics_table2.json \
    --wandb-group '$WANDB_GROUP' --wandb-run-name '${WANDB_NAME}_metrics' && \
  bash scripts/export_all_metrics.sh '$WANDB_GROUP' && \
  bash scripts/plot_from_metrics_csv.py --experiment-tag table2 && \
  echo '=== Table 2 complete ==='; bash"

exp_tmux_start "$SESSION" table2 "$CMD 2>&1 | tee '$LOG'"
exp_print_launch_summary "$SESSION" \
  "GPUs: $GPUS | restarts: $RESTARTS" \
  "epochs: $EPOCHS | batch: $BATCH_SIZE | preset: $EXP_PRESET" \
  "output: $OUTPUT_DIR" \
  "log: $LOG" \
  "W&B: $WANDB_NAME (group $WANDB_GROUP)"
