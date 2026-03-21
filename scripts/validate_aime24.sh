#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT_DIR/.venv/bin/lm-eval" validate \
  --include_path "$ROOT_DIR/tasks" \
  --tasks aime24_sc_25
