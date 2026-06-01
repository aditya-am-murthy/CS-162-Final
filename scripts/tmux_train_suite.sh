#!/usr/bin/env bash
# Launch each model training in its own tmux window (Colab SSH or remote GPU box).
# Usage: bash scripts/tmux_train_suite.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="cs162-cartography"
cd "$ROOT"

if ! command -v tmux &>/dev/null; then
  echo "tmux not found; run sequentially:"
  exec python scripts/train_all_models.py "$@"
fi

tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"

PRESETS=(distilbert roberta-base llama-3.2-1b ministral-3b)
tmux new-session -d -s "$SESSION" -n "${PRESETS[0]}"
tmux send-keys -t "$SESSION:0" "cd '$ROOT' && python scripts/run_cartography_experiment.py --task dynamic --preset ${PRESETS[0]} --wandb-run-name snli_${PRESETS[0]}" C-m

for i in "${!PRESETS[@]}"; do
  [[ $i -eq 0 ]] && continue
  p="${PRESETS[$i]}"
  tmux new-window -t "$SESSION" -n "$p"
  tmux send-keys -t "$SESSION:$i" "cd '$ROOT' && python scripts/run_cartography_experiment.py --task dynamic --preset $p --wandb-run-name snli_$p" C-m
done

echo "attached session: tmux attach -t $SESSION"
echo "list windows: tmux list-windows -t $SESSION"
