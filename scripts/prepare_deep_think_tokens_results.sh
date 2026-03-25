#!/bin/bash

set -euo pipefail

uv run python -m src.deep_think_tokens_project.prepare_results "$@"
