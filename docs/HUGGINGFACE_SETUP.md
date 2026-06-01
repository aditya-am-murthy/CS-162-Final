# Hugging Face setup (Colab + local)

## 1. Account and access token

1. Create an account at [huggingface.co](https://huggingface.co).
2. Open **[Settings → Access Tokens](https://huggingface.co/settings/tokens)**.
3. Create a token with **Read** permission (enough to download models and datasets).
4. Copy the token (starts with `hf_`).

## 2. Log in on your machine

**Option A — file (matches this repo’s W&B pattern)**

```bash
cp hf_credentials.example.txt hf_credentials.txt
# edit hf_token=hf_...
```

Training scripts load this via `HF_TOKEN` automatically when present.

**Option B — CLI**

```bash
pip install huggingface_hub
huggingface-cli login
# paste token when prompted
```

**Option C — environment variable**

```bash
export HF_TOKEN=hf_your_token_here
```

**Google Colab**

```python
from huggingface_hub import login
login(token="hf_...")  # or use Colab secret HF_TOKEN
```

Or in a cell before training:

```bash
!huggingface-cli login --token hf_...
```

## 3. Accept model licenses (gated models)

Some weights require clicking **Agree** on the model page while logged in. Your token only works after that.

| Model in this project | Gated? | Action |
|----------------------|--------|--------|
| `unsloth/Ministral-3-3B-Base-2512-unsloth-bnb-4bit` | **No** | Download with token only |
| `mistralai/Ministral-3-3B-Base-2512` (parent) | Check HF page | Accept if prompted |
| `meta-llama/Llama-3.2-1B` | **Yes** | Accept [Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| `stanfordnlp/snli` (dataset) | Usually open | No extra step |

For **Ministral 3**, the Unsloth 4-bit repo is Apache 2.0 and not gated; you still need a token for reliable downloads and rate limits.

## 4. Extra packages for Ministral 3 / Unsloth

The Unsloth checkpoint is **already 4-bit** — do not enable a second `load_in_4bit` in training configs.

```bash
pip install -r requirements-train.txt
pip install unsloth  # recommended loader for unsloth/* checkpoints
# optional if using native Mistral 3 stack:
# pip install "transformers>=5.0.0rc0" "mistral-common>=1.8.6"
```

Preset: `--preset ministral-3b` → `unsloth/Ministral-3-3B-Base-2512-unsloth-bnb-4bit`

## 5. Verify login

```bash
python scripts/test_hf_credentials.py
```

Expected: `OK: authenticated as <your_username>` and a small config download test.

## 6. Common errors

| Error | Fix |
|-------|-----|
| `401 Unauthorized` / `Invalid user token` from **HF_TOKEN environment variable** | A stale `HF_TOKEN` in your shell overrides login. Run `unset HF_TOKEN HUGGING_FACE_HUB_TOKEN`, put a fresh token in `hf_credentials.txt`, re-run the test. Check `grep HF_TOKEN ~/.zshrc`. |
| `401` with no env mention | Create a new Read token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `403 / gated repo` | Open model page in browser → accept license → retry |
| `CUDA OOM` | Lower `--max-train-samples`, `--batch-size`, or use `--preset distilbert` for smoke tests |
| `unsloth not found` | `pip install unsloth` or install transformers stack from Ministral model card |

## 7. What you do *not* need

- A **Write** token unless you push models to your own HF hub.
- Separate Mistral API keys (console.mistral.ai) — only for their hosted API, not HF weights.
