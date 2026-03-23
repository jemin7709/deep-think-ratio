#!/bin/bash

# run_dir 하나만 넘기면 JSD matrix, heatmap, DTR JSON을 순서대로 만든다.
RUN_DIR="${1:?usage: ./run_dtr.sh results/<task>/<model>/<seed>/<timestamp>}"

uv run python -m src.dtr.jsd "$RUN_DIR" --save-heatmap
uv run python -m src.dtr.dtr "$RUN_DIR"
