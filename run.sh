#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

uv run run.py \
--task-config tasks/aime24/aime24_custom.yaml \
--model-config models/gpt-oss-120b.yaml \
--seed 0 --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 --seed 6 --seed 7 --seed 8 --seed 9 \
