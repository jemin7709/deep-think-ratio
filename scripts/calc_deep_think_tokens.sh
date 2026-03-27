#!/bin/bash

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <run_dir>" >&2
  exit 1
fi

RUN_DIR="$1"

uv run python -m src.deep_think_tokens_project.jsd "$RUN_DIR"
uv run python -m src.deep_think_tokens_project.dtr "$RUN_DIR"
