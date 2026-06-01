#!/usr/bin/env python3
"""Smoke test: Hugging Face token and optional Ministral config download."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.common import DEFAULT_HF_CREDENTIALS_PATH, load_hf_credentials, resolve_hf_token


def _print_401_help() -> None:
    print("\n401 = token rejected by Hugging Face. Try:")
    print("  1. New Read token: https://huggingface.co/settings/tokens")
    print(f"  2. Save as {DEFAULT_HF_CREDENTIALS_PATH}:")
    print("       hf_token=hf_xxxxxxxx")
    print("  3. Remove a bad shell token (often the real culprit):")
    print("       unset HF_TOKEN HUGGING_FACE_HUB_TOKEN")
    print("       # check: grep HF_TOKEN ~/.zshrc ~/.bashrc")
    print("  4. Re-run: python scripts/test_hf_credentials.py")


def main() -> None:
    shell_hf = os.environ.get("HF_TOKEN")
    creds = load_hf_credentials()
    token, source = resolve_hf_token()
    if not token:
        print("no token found.")
        print(f"  create {DEFAULT_HF_CREDENTIALS_PATH} from hf_credentials.example.txt")
        print("  or: huggingface-cli login")
        sys.exit(1)

    print(f"token source: {source} (length {len(token)})")
    if shell_hf and source == "file" and shell_hf != token:
        print("  (ignored stale HF_TOKEN from your shell — good)")

    from huggingface_hub import whoami
    from huggingface_hub.errors import HfHubHTTPError

    try:
        info = whoami(token=token)
    except HfHubHTTPError as e:
        if "401" in str(e) or "Invalid" in str(e):
            _print_401_help()
        else:
            print(f"error: {e}")
        sys.exit(1)

    print(f"OK: authenticated as {info.get('name', info)}")

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="unsloth/Ministral-3-3B-Base-2512-unsloth-bnb-4bit",
            filename="config.json",
            token=token,
        )
        print(f"OK: downloaded config -> {path}")
    except Exception as e:
        print(f"warn: config download failed ({e})")


if __name__ == "__main__":
    main()
