#!/bin/bash

RUN_DIR="$1"

uv run python -m src.experiment.think_n "$RUN_DIR" \
  --prefix-len 50 \
  --top-fraction 0.5