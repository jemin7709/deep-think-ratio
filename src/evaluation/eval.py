from __future__ import annotations

import argparse
from pathlib import Path

from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker

from src.evaluation.common import (
    build_evaluator_config,
    load_model_settings,
    load_task_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lm-eval generation for one task/model pair and save raw artifacts."
    )
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--limit", type=float)
    return parser.parse_args()


def save_raw_results(results: dict, tracker: EvaluationTracker) -> dict[str, list[dict]]:
    samples = results.pop("samples")
    tracker.save_results_aggregated(results=results, samples=samples)
    for task_name, task_samples in samples.items():
        tracker.save_results_samples(task_name=task_name, samples=task_samples)
    return samples


def validate_repeats(
    samples_by_task: dict[str, list[dict]],
    *,
    task_name: str,
    expected_repeats: int,
) -> None:
    task_samples = samples_by_task.get(task_name, [])
    if not task_samples:
        raise ValueError(f"no samples were produced for task {task_name}")
    for sample in task_samples:
        completions = [str(response) for response in sample["resps"][0]]
        actual_repeats = len(completions)
        if actual_repeats != expected_repeats:
            raise ValueError(
                f"expected {expected_repeats} completions per problem, "
                f"found {actual_repeats} for doc_id={sample['doc_id']}"
            )


def run_evaluation(
    *,
    task_config_path: Path,
    model_config_path: Path,
    seed: int,
    run_dir: Path,
    limit: float | None = None,
) -> Path:
    task_settings = load_task_settings(task_config_path)
    model_settings = load_model_settings(model_config_path)
    config = build_evaluator_config(
        task_settings,
        model_settings,
        run_dir=run_dir,
        seed=seed,
        limit=limit,
    )
    tracker = EvaluationTracker(output_path=config.output_path)
    task_manager = config.process_tasks(config.metadata)

    results = simple_evaluate(
        model=config.model,
        model_args=config.model_args,
        tasks=config.tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        max_batch_size=config.max_batch_size,
        device=config.device,
        use_cache=config.use_cache,
        cache_requests=config.cache_requests.get("cache_requests", False),
        rewrite_requests_cache=config.cache_requests.get(
            "rewrite_requests_cache", False
        ),
        delete_requests_cache=config.cache_requests.get(
            "delete_requests_cache", False
        ),
        limit=config.limit,
        samples=config.samples,
        check_integrity=config.check_integrity,
        write_out=config.write_out,
        log_samples=True,
        evaluation_tracker=tracker,
        system_instruction=config.system_instruction,
        apply_chat_template=config.apply_chat_template,
        fewshot_as_multiturn=config.fewshot_as_multiturn,
        gen_kwargs=config.gen_kwargs,
        task_manager=task_manager,
        verbosity=config.verbosity,
        predict_only=config.predict_only,
        random_seed=config.seed[0],
        numpy_random_seed=config.seed[1],
        torch_random_seed=config.seed[2],
        fewshot_random_seed=config.seed[3],
        confirm_run_unsafe_code=config.confirm_run_unsafe_code,
        metadata=config.metadata,
    )
    if results is None:
        raise RuntimeError("lm-eval returned no results")

    samples = save_raw_results(results, tracker)
    validate_repeats(
        samples,
        task_name=task_settings.name,
        expected_repeats=task_settings.repeats,
    )
    return run_dir


def main() -> None:
    args = parse_args()
    run_evaluation(
        task_config_path=args.task_config,
        model_config_path=args.model_config,
        seed=args.seed,
        run_dir=args.run_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
