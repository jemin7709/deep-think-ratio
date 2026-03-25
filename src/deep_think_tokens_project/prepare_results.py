"""원본 raw run 산출물만 별도 실험 루트로 복사한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.deep_think_tokens_project.io import DEFAULT_SOURCE_ROOT
from src.deep_think_tokens_project.io import DEFAULT_TARGET_ROOT
from src.deep_think_tokens_project.io import copy_raw_run
from src.deep_think_tokens_project.io import discover_run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy only raw run artifacts into results_deep_think_tokens."
    )
    parser.add_argument("source_root", nargs="?", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("target_root", nargs="?", type=Path, default=DEFAULT_TARGET_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    target_root = args.target_root.resolve()
    run_dirs = discover_run_dirs(source_root)
    print(f"source_root: {source_root}")
    print(f"target_root: {target_root}")
    print(f"복사할 run 수: {len(run_dirs)}")
    for index, run_dir in enumerate(run_dirs, start=1):
        copied = copy_raw_run(
            run_dir,
            source_root=source_root,
            target_root=target_root,
        )
        print(f"[{index}/{len(run_dirs)}] copied {run_dir} -> {copied}")


if __name__ == "__main__":
    main()
