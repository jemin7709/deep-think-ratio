#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_NAME="${1:?usage: scripts/run_aime24.sh <model> [--n N] [--k K] [--limit L]}"
shift || true

N=25
K=1
LIMIT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n)
      N="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

MODEL_CONFIG="$ROOT_DIR/models/${MODEL_NAME}.yaml"
if [[ ! -f "$MODEL_CONFIG" ]]; then
  echo "missing model config: $MODEL_CONFIG" >&2
  exit 1
fi

if (( K > N )); then
  echo "--k must be <= --n (got k=$K n=$N)" >&2
  exit 1
fi

TIMESTAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_DIR="$ROOT_DIR/results/$MODEL_NAME/$TIMESTAMP"
mkdir -p "$RUN_DIR"

RUN_ARGS=(
  --config "$MODEL_CONFIG"
  --output-path "$RUN_DIR"
  --n "$N"
)

if [[ -n "$LIMIT" ]]; then
  RUN_ARGS+=(--limit "$LIMIT")
fi

"$ROOT_DIR/.venv/bin/python" -m tasks.aime24.run "${RUN_ARGS[@]}"

"$ROOT_DIR/.venv/bin/python" -m tasks.aime24.metrics --run-dir "$RUN_DIR" --n "$N" --k "$K"
