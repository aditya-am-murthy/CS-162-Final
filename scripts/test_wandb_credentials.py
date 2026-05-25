#!/usr/bin/env python3
"""Verify wandb_credentials.txt loads and authenticates with Weights & Biases."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import wandb
from scripts.common import DEFAULT_CREDENTIALS_PATH, load_wandb_credentials


def main() -> int:
    creds = load_wandb_credentials()
    missing = [k for k in ("api_key", "entity", "project") if k not in creds]
    if missing:
        print(f"FAIL: missing in {DEFAULT_CREDENTIALS_PATH}: {', '.join(missing)}")
        return 1

    print(f"loaded credentials from {DEFAULT_CREDENTIALS_PATH}")
    print(f"  entity:  {creds['entity']}")
    print(f"  project: {creds['project']}")
    print(f"  api_key: {'*' * 8}...{creds['api_key'][-4:]}")

    api = wandb.Api()
    viewer = api.viewer
    username = getattr(viewer, "username", None) or getattr(viewer, "entity", None) or str(viewer)
    print(f"OK: authenticated as {username}")

    run = wandb.init(
        project=creds["project"],
        entity=creds["entity"],
        job_type="credentials_test",
        name="credentials_test",
        config={"test": True},
    )
    wandb.log({"credentials_test": 1.0})
    url = run.url
    wandb.finish()

    print(f"OK: test run logged successfully")
    print(f"  run url: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
