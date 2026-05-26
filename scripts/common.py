"""Shared CLI helpers for experiment scripts."""

from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CACHE_DIR = _REPO_ROOT / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency for local runs
    wandb = None

DEFAULT_CREDENTIALS_PATH = _REPO_ROOT / "wandb_credentials.txt"


def load_wandb_credentials(
    credentials_path: Optional[Path] = None,
) -> Dict[str, str]:
    """
    Read api_key, entity, and project from wandb_credentials.txt.

    Lines use key=value. Lines starting with # are ignored.
    """
    path = credentials_path or DEFAULT_CREDENTIALS_PATH
    creds: Dict[str, str] = {}
    if not path.is_file():
        return creds

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip().lower(), value.strip()
            if value and value not in (
                "YOUR_WANDB_API_KEY_HERE",
                "YOUR_WANDB_USERNAME_OR_TEAM",
            ):
                creds[key] = value

    if creds.get("api_key"):
        os.environ["WANDB_API_KEY"] = creds["api_key"]
    return creds


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wandb-project",
        default="cs162-dataset-cartography",
        help="Weights & Biases project name",
    )
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional W&B entity (team or user)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help="Path to wandb_credentials.txt",
    )


def load_pipeline_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_wandb(
    args: argparse.Namespace,
    job_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[wandb.sdk.wandb_run.Run]:
    if not use_wandb(args):
        return None

    creds_path = getattr(args, "wandb_credentials", DEFAULT_CREDENTIALS_PATH)
    creds = load_wandb_credentials(creds_path)

    project = args.wandb_project or creds.get("project", "cs162-dataset-cartography")
    entity = args.wandb_entity or creds.get("entity")

    return wandb.init(
        project=project,
        name=args.wandb_run_name or job_type,
        entity=entity,
        job_type=job_type,
        config=config or {},
    )


def finish_wandb() -> None:
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def use_wandb(args: argparse.Namespace) -> bool:
    return (not getattr(args, "no_wandb", False)) and wandb is not None
