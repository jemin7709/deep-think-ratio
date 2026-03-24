#!/bin/bash

set -euo pipefail

# run_dir 하나만 넘기면 JSD matrix와 DTR JSON을 순서대로 만든다.
if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <run_dir>" >&2
  exit 1
fi

RUN_DIR="$1"

uv run python -m src.dtr.jsd "$RUN_DIR"
uv run python -m src.dtr.dtr "$RUN_DIR"
