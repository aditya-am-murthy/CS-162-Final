#!/usr/bin/env bash
# Run once in Google Colab (Runtime → T4 GPU) after cloning the repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

pip install -q -r requirements.txt -r requirements-train.txt
pip install -q bitsandbytes accelerate

# cuda wheel if colab pytorch is cpu-only (usually colab has cuda)
python -c "import torch; print('cuda:', torch.cuda.is_available())"

if [[ ! -f wandb_credentials.txt ]]; then
  cp wandb_credentials.example.txt wandb_credentials.txt
  echo "edit wandb_credentials.txt with api_key, entity, project"
fi
if [[ ! -f hf_credentials.txt ]]; then
  cp hf_credentials.example.txt hf_credentials.txt
  echo "edit hf_credentials.txt with hf_token=hf_..."
fi
pip install -q unsloth huggingface_hub

echo "smoke test (distilbert, small):"
python scripts/run_cartography_experiment.py \
  --task snli --preset distilbert \
  --max-train-samples 500 --epochs 2 \
  --wandb-run-name colab_smoke

echo "full suite:"
echo "  python scripts/train_all_models.py --max-train-samples 10000 --epochs 5"
