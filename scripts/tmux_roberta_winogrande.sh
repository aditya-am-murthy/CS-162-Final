#!/usr/bin/env bash
# Paper-style static data map: RoBERTa-base on WinoGrande (NOT Idea #2 dynamic).
#
# Usage:
#   bash scripts/tmux_roberta_winogrande.sh
#   tmux attach -t cs162-winogrande
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="cs162-winogrande"
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

# --task snli = static cartography map (no per-epoch curriculum / Idea #2 movement)
CMD="cd '$ROOT' && $PYTHON scripts/run_cartography_experiment.py \\
  --task snli \\
  --dataset winogrande \\
  --preset roberta-base \\
  --max-train-samples 5000 \\
  --max-eval-samples 1000 \\
  --epochs 5 \\
  --wandb-run-name winogrande_roberta_paper_5k"

tmux new-session -d -s "$SESSION" -n roberta-winogrande
tmux send-keys -t "$SESSION:0" "$CMD" C-m

echo "Started tmux session: $SESSION"
echo "  attach:  tmux attach -t $SESSION"
echo "  W&B run: winogrande_roberta_paper_5k"
echo "  task=snli (static map), dataset=winogrande, 5000 train / 5 epochs"
