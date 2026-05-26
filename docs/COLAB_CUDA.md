# Colab CUDA troubleshooting

## Symptom

`colab_setup.sh` prints `cuda: False` or training shows `torch 2.x.x+cpu` and ~20s/step.

## Cause

`pip install -r requirements-train.txt` installs **PyPI `torch`**, which is often **CPU-only** and **overwrites** Colab’s preinstalled CUDA build. `unsloth` can do the same.

## Fix (Colab)

1. **Runtime → Change runtime type → T4 GPU**
2. **Runtime → Restart session** (required after changing GPU)
3. Re-run clone cell, then:

```bash
SKIP_SMOKE=1 bash scripts/colab_setup.sh
```

4. Verify:

```bash
python scripts/check_cuda.py
```

Expected: `cuda.is_available()=True` and a GPU name (e.g. Tesla T4).

## One-liner emergency repair

If you already ran the old setup and have `+cpu` torch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python scripts/check_cuda.py
```

## Speed expectation

| Device | ~500 samples, DistilBERT, 2 epochs |
|--------|-------------------------------------|
| T4 GPU | ~15–40 min total |
| CPU | hours |
