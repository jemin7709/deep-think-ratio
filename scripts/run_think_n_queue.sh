#!/bin/bash

set -euo pipefail

# GPU 하나가 맡을 run 디렉터리 목록을 순서대로 처리한다.
# Think@n은 jsd_matrices만 있으면 되므로, 먼저 모든 run의 JSD/DTR을 끝내고
# 그 다음 Think@n, 마지막에 heatmap만 별도 루프로 렌더링한다.

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <gpu_id> <run_dir> [<run_dir> ...]" >&2
  exit 1
fi

GPU_ID="$1"
shift

export CUDA_VISIBLE_DEVICES="$GPU_ID"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

for run_dir in "$@"; do
  echo "[$(timestamp)] gpu=$GPU_ID dtr-start run_dir=$run_dir"
  bash scripts/calc_dtr.sh "$run_dir"
  echo "[$(timestamp)] gpu=$GPU_ID dtr-done run_dir=$run_dir"
done

for run_dir in "$@"; do
  echo "[$(timestamp)] gpu=$GPU_ID think-n-start run_dir=$run_dir"
  bash scripts/calc_think_n.sh "$run_dir"
  echo "[$(timestamp)] gpu=$GPU_ID think-n-done run_dir=$run_dir"
done

for run_dir in "$@"; do
  echo "[$(timestamp)] gpu=$GPU_ID think-n-bottom-start run_dir=$run_dir"
  bash scripts/calc_think_n_bottom.sh "$run_dir"
  echo "[$(timestamp)] gpu=$GPU_ID think-n-bottom-done run_dir=$run_dir"
done

for run_dir in "$@"; do
  echo "[$(timestamp)] gpu=$GPU_ID heatmap-start run_dir=$run_dir"
  bash scripts/calc_heatmaps.sh "$run_dir"
  echo "[$(timestamp)] gpu=$GPU_ID heatmap-done run_dir=$run_dir"
done

echo "[$(timestamp)] gpu=$GPU_ID queue complete"
