#!/bin/bash

# run_dir 하나만 넘기면 기존 JSD matrix 캐시에서 heatmap PNG만 다시 만든다.
RUN_DIR="$1"

uv run python -m src.dtr.jsd "$RUN_DIR" --render-heatmaps-only
