#!/usr/bin/env python3
"""Diagnose PyTorch CUDA on Colab/local GPU. Exit 1 if CUDA unavailable."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="check", help="Stage name for logging")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only set exit code (0=cuda ok, 1=not ok)",
    )
    args = parser.parse_args()

    import torch

    version = torch.__version__
    cuda_ok = torch.cuda.is_available()
    is_cpu_wheel = "+cpu" in version.lower()

    if not args.quiet:
        print(f"[{args.label}] torch={version}  cuda.is_available()={cuda_ok}")
        if is_cpu_wheel:
            print("  problem: CPU-only torch wheel (+cpu). Training will be very slow.")
            print("  fix on Colab: bash scripts/colab_setup.sh  OR")
            print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
        if cuda_ok:
            print(f"  gpu: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  vram: {props.total_memory / (1024**3):.1f} GB")
        else:
            print("  fix: Colab → Runtime → Change runtime type → T4 GPU → Restart session")

    sys.exit(0 if cuda_ok else 1)


if __name__ == "__main__":
    main()
