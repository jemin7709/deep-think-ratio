from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.common import build_run_dir, load_model_settings, load_task_settings
from src.evaluation.eval import run_evaluation
from tasks.aime24.metrics import write_postprocess_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and postprocess one task across many models and seeds."
    )
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, action="append", type=Path)
    parser.add_argument("--seed", required=True, action="append", type=int)
    parser.add_argument("--limit", type=float)
    return parser.parse_args()


def run_one(
    *,
    task_config_path: Path,
    model_config_path: Path,
    seed: int,
    limit: float | None,
) -> Path:
    task_settings = load_task_settings(task_config_path)
    model_settings = load_model_settings(model_config_path)
    run_dir = build_run_dir(task_settings.name, model_settings.name, seed)
    try:
        run_evaluation(
            task_config_path=task_config_path,
            model_config_path=model_config_path,
            seed=seed,
            run_dir=run_dir,
            limit=limit,
        )
        write_postprocess_artifacts(run_dir=run_dir)
    except BaseException:
        if run_dir.exists() and not any(run_dir.iterdir()):
            run_dir.rmdir()
        raise
    return run_dir


def main() -> None:
    args = parse_args()
    for model_config_path in args.model_config:
        for seed in args.seed:
            run_one(
                task_config_path=args.task_config,
                model_config_path=model_config_path,
                seed=seed,
                limit=args.limit,
            )


if __name__ == "__main__":
    main()
