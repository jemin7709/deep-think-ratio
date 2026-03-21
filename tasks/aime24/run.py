from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from lm_eval import simple_evaluate
from lm_eval.config.evaluate_config import EvaluatorConfig
from lm_eval.loggers import EvaluationTracker
from lm_eval.tasks import get_task_dict


TASK_NAME = "aime24_sc_25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AIME24 with a runtime-configured repeat count."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--n", default=25, type=int)
    parser.add_argument("--limit", default=None, type=float)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvaluatorConfig:
    model_config = EvaluatorConfig.load_yaml_config(args.config)
    cfg = EvaluatorConfig(
        tasks=[TASK_NAME],
        output_path=args.output_path,
        include_path=str(ROOT_DIR / "tasks"),
        log_samples=True,
        limit=args.limit,
        **model_config,
    )
    return cfg._parse_dict_args()._configure()


def save_results(results: dict, tracker: EvaluationTracker) -> None:
    samples = results.pop("samples")
    tracker.save_results_aggregated(results=results, samples=samples)
    for task_name in results["configs"]:
        tracker.save_results_samples(task_name=task_name, samples=samples[task_name])


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    tracker = EvaluationTracker(output_path=cfg.output_path)

    task_manager = cfg.process_tasks(cfg.metadata)
    task = get_task_dict(cfg.tasks, task_manager)[TASK_NAME]
    task.set_config("repeats", args.n)

    results = simple_evaluate(
        model=cfg.model,
        model_args=cfg.model_args,
        tasks=[task],
        num_fewshot=cfg.num_fewshot,
        batch_size=cfg.batch_size,
        max_batch_size=cfg.max_batch_size,
        device=cfg.device,
        use_cache=cfg.use_cache,
        cache_requests=cfg.cache_requests.get("cache_requests", False),
        rewrite_requests_cache=cfg.cache_requests.get("rewrite_requests_cache", False),
        delete_requests_cache=cfg.cache_requests.get("delete_requests_cache", False),
        limit=cfg.limit,
        samples=cfg.samples,
        check_integrity=cfg.check_integrity,
        write_out=cfg.write_out,
        log_samples=True,
        evaluation_tracker=tracker,
        system_instruction=cfg.system_instruction,
        apply_chat_template=cfg.apply_chat_template,
        fewshot_as_multiturn=cfg.fewshot_as_multiturn,
        gen_kwargs=cfg.gen_kwargs,
        task_manager=task_manager,
        verbosity=cfg.verbosity,
        predict_only=cfg.predict_only,
        random_seed=cfg.seed[0] if cfg.seed else None,
        numpy_random_seed=cfg.seed[1] if cfg.seed else None,
        torch_random_seed=cfg.seed[2] if cfg.seed else None,
        fewshot_random_seed=cfg.seed[3] if cfg.seed else None,
        confirm_run_unsafe_code=cfg.confirm_run_unsafe_code,
        metadata=cfg.metadata,
    )

    if results is not None:
        save_results(results, tracker)


if __name__ == "__main__":
    main()
