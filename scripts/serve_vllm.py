from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.common import build_vllm_command, load_model_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run vllm serve using the server section of a model config."
    )
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_settings = load_model_settings(args.model_config)
    command = build_vllm_command(model_settings, seed=args.seed)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
