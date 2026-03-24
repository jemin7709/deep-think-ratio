#!/bin/bash

RUN_DIR="$1"

uv run python -m src.experiment.think_n_bottom "$RUN_DIR" \
  --prefix-len 50 \
  --bottom-fraction 0.5
