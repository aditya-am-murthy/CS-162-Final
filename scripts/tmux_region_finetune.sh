#!/usr/bin/env bash
# §3 region subset retraining: WinoGrande RoBERTa-large, parallel GPUs + W&B.
#
# Usage:
#   bash scripts/tmux_region_finetune.sh
#   tmux attach -t cs162-region-finetune
#
# Override GPUs or coordinates:
#   GPUS=0,1,2,3 bash scripts/tmux_region_finetune.sh
#   INPUT=path/to/cartography_with_regions.jsonl bash scripts/tmux_region_finetune.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${SESSION:-cs162-region-finetune}"
CONDA_ENV="${CONDA_ENV:-cs162-cartography}"
PYTHON="${PYTHON:-}"

GPUS="${GPUS:-0,1,2,3,4}"
INPUT="${INPUT:-$ROOT/results/20260609_074628_snli_winogrande_roberta-large/fixed-maps/adaptive/20260609_074628_cartography_with_regions.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data/processed/region_finetune_winogrande}"
WANDB_NAME="${WANDB_NAME:-winogrande_33pct_roberta_large}"
RESTARTS="${RESTARTS:-1}"

if [[ -z "$PYTHON" ]]; then
  if command -v conda &>/dev/null; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    PYTHON="$(which python)"
  else
    PYTHON="python"
  fi
fi

cd "$ROOT"

if [[ ! -f "$INPUT" ]]; then
  echo "missing coordinates input: $INPUT" >&2
  echo "set INPUT= to cartography_with_regions.jsonl from your WinoGrande map run" >&2
  exit 1
fi

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

CMD="cd '$ROOT' && $PYTHON scripts/09_region_finetune.py \\
  --train \\
  --dataset winogrande \\
  --preset roberta-large \\
  --epochs 6 \\
  --batch-size 64 \\
  --restarts $RESTARTS \\
  --gpus $GPUS \\
  --input '$INPUT' \\
  --output-dir '$OUTPUT_DIR' \\
  --wandb-run-name $WANDB_NAME"

tmux new-session -d -s "$SESSION" -n region-finetune
tmux send-keys -t "$SESSION:0" "$CMD" C-m

echo "Started tmux session: $SESSION"
echo "  attach:  tmux attach -t $SESSION"
echo "  GPUs:    $GPUS (5 parallel workers by default)"
echo "  input:   $INPUT"
echo "  output:  $OUTPUT_DIR"
echo "  W&B:     metrics only; charts saved locally under training_runs/*/figures/"
echo "  restarts: $RESTARTS (set RESTARTS=3 for paper-style 3-seed averaging)"
echo "  jobs:    8 strategies x $RESTARTS restarts = $((8 * RESTARTS)) training runs"
