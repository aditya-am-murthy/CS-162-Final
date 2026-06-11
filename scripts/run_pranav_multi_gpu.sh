#!/usr/bin/env bash
# Launch Pranav scripts 08 + 12 with wandb, multi-GPU, and per-epoch progress bars.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cs162-cartography
mkdir -p logs

INPUT="${INPUT:-results/20260609_074628_snli_winogrande_roberta-large/dynamics/cartography_with_regions.jsonl}"
GPUS_08="${GPUS_08:-1,2}"
GPU_12="${GPU_12:-0}"
WANDB_GROUP="${WANDB_GROUP:-pranav-scripts}"
SESSION="${TMUX_SESSION:-cs162-pranav}"
export PYTHONUNBUFFERED=1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists — attach with: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n script12 \
  "cd '$ROOT' && source '$HOME/miniconda3/etc/profile.d/conda.sh' && conda activate cs162-cartography && \
   export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$GPU_12' && \
   python scripts/12_noise_detection_paper.py \
     --train \
     --input '$INPUT' \
     --wandb-run-name pranav_12_noise_detection \
     --wandb-group '$WANDB_GROUP' \
   2>&1 | tee logs/pranav_12_train.log; \
   echo; echo '=== script 12 finished ==='; bash"

tmux new-window -t "$SESSION" -n script08 \
  "cd '$ROOT' && source '$HOME/miniconda3/etc/profile.d/conda.sh' && conda activate cs162-cartography && \
   export PYTHONUNBUFFERED=1 && \
   python scripts/08_role_easy_to_learn.py \
     --train \
     --input '$INPUT' \
     --dataset winogrande \
     --preset roberta-base \
     --gpus '$GPUS_08' \
     --wandb-run-name pranav_08_easy_to_learn \
     --wandb-group '$WANDB_GROUP' \
   2>&1 | tee logs/pranav_08_train.log; \
   echo; echo '=== script 08 finished ==='; bash"

echo "Started tmux session: $SESSION"
echo "  3 GPUs total: cuda:${GPU_12} (script 12) + gpus:${GPUS_08} (script 08)"
echo "  window script12 -> cuda:${GPU_12}  (12_noise_detection_paper.py --train)"
echo "  window script08 -> gpus:${GPUS_08} (08_role_easy_to_learn.py --train, epoch progress bar)"
echo ""
echo "Attach:  tmux attach -t $SESSION"
echo "Logs:    logs/pranav_12_train.log  logs/pranav_08_train.log"
