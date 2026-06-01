#!/usr/bin/env bash
# Google Colab: Runtime → T4 GPU → Restart session → run this script.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== 1. GPU runtime ==="
if ! command -v nvidia-smi &>/dev/null; then
  echo "FATAL: nvidia-smi not found."
  echo "  Colab: Runtime → Change runtime type → T4 GPU → Restart session"
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo ""
echo "=== 2. torch BEFORE pip (should be CUDA on Colab) ==="
python scripts/check_cuda.py --label before_pip || true

echo ""
echo "=== 3. install deps (never torch from PyPI on Colab) ==="
# requirements-train.txt includes torch>=2.1 — that installs +cpu and breaks Colab.
pip install -q -r requirements.txt -r requirements-colab.txt

echo ""
echo "=== 4. verify CUDA after core deps ==="
if ! python scripts/check_cuda.py --quiet; then
  echo "CUDA lost after pip — reinstalling PyTorch cu124 wheel..."
  pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  python scripts/check_cuda.py --label after_torch_reinstall
fi

echo ""
echo "=== 5. unsloth (optional; can replace torch — we re-fix if needed) ==="
if pip install -q unsloth 2>/dev/null; then
  if ! python scripts/check_cuda.py --quiet; then
    echo "unsloth replaced torch with CPU build — reinstalling cu124..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    python scripts/check_cuda.py --label after_unsloth_fix
  fi
else
  echo "warn: unsloth install failed (ministral-3b preset may need it later)"
fi

if [[ ! -f wandb_credentials.txt ]]; then
  cp wandb_credentials.example.txt wandb_credentials.txt
  echo "edit wandb_credentials.txt"
fi
if [[ ! -f hf_credentials.txt ]]; then
  cp hf_credentials.example.txt hf_credentials.txt
  echo "edit hf_credentials.txt"
fi

python scripts/check_cuda.py --label final

if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
  echo ""
  echo "=== 6. smoke test on GPU ==="
  python scripts/run_cartography_experiment.py \
    --task snli --preset distilbert \
    --max-train-samples 500 --epochs 2 \
    --wandb-run-name colab_smoke
fi

echo ""
echo "OK. Full suite:"
echo "  python scripts/train_all_models.py --max-train-samples 10000 --epochs 5"
