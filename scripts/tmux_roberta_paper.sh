#!/usr/bin/env bash
# Paper-style static cartography: RoBERTa on one GLUE dataset, pinned to one GPU.
#
# Usage:
#   bash scripts/tmux_roberta_paper.sh <gpu_id> <dataset> <tmux_session> <wandb_run_name>
#
# Examples:
#   bash scripts/tmux_roberta_paper.sh 1 mnli cs162-mnli mnli_roberta_paper_3k
#   bash scripts/tmux_roberta_paper.sh 2 qnli cs162-qnli qnli_roberta_paper_3k
set -euo pipefail

GPU_ID="${1:?gpu_id required (e.g. 1)}"
DATASET="${2:?dataset required (snli|mnli|qnli|winogrande)}"
SESSION="${3:?tmux session name required}"
WANDB_NAME="${4:?wandb run name required}"
MAX_TRAIN="${5:-3000}"
MAX_EVAL="${6:-600}"
EPOCHS="${7:-5}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-cs162-cartography}"
PYTHON="${PYTHON:-}"

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

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

CMD="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON scripts/run_cartography_experiment.py \\
  --task snli \\
  --dataset $DATASET \\
  --preset roberta-base \\
  --max-train-samples $MAX_TRAIN \\
  --max-eval-samples $MAX_EVAL \\
  --epochs $EPOCHS \\
  --wandb-run-name $WANDB_NAME"

tmux new-session -d -s "$SESSION" -n "roberta-$DATASET"
tmux send-keys -t "$SESSION:0" "$CMD" C-m

echo "Started tmux session: $SESSION (cuda:$GPU_ID)"
echo "  attach:  tmux attach -t $SESSION"
echo "  dataset: $DATASET | train=$MAX_TRAIN | epochs=$EPOCHS"
echo "  W&B run: $WANDB_NAME"
