#!/usr/bin/env bash
# Create or update the cs162-cartography conda environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. install Miniconda/Anaconda first:"
  echo "  https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# accept default-channel ToS on fresh Miniconda installs (non-interactive)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if conda env list | awk '{print $1}' | grep -qx cs162-cartography; then
  echo "updating existing env: cs162-cartography"
  conda env update -f environment.yml --prune
else
  echo "creating env: cs162-cartography"
  conda env create -f environment.yml
fi

echo ""
echo "done. activate with:"
echo "  conda activate cs162-cartography"
echo ""
echo "optional GPU training deps:"
echo "  pip install -r requirements-train.txt"
echo ""
echo "then run experiments:"
echo "  python scripts/run_all_experiments.py --wandb-project cs162-dataset-cartography"
echo "  python scripts/train_and_collect_dynamics.py --preset distilbert"
