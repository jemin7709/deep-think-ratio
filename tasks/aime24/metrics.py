from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import stdev

from src.evaluation.common import find_task_config_path, load_task_settings

from .utils import (
    resolve_model_identity,
    resolve_reasoning_tags,
    score_avg_at_n,
    score_maj_at_n,
    score_pass_at_k,
)


TASK_NAME = "aime24_custom"


def sample_stddev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pass@k, avg@n, and maj@n from an lm-eval AIME24 run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task")
    parser.add_argument("--k", default=1, type=int)
    return parser.parse_args()


def latest_file(run_dir: Path, pattern: str) -> Path:
    matches = list(run_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no files matched {pattern} under {run_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_aggregated(run_dir: Path) -> dict:
    return json.loads(latest_file(run_dir, "results_*.json").read_text())


def infer_task_name(aggregated: dict) -> str:
    for key in (TASK_NAME,):
        if key in aggregated.get("results", {}):
            return key
        if key in aggregated.get("configs", {}):
            return key

    for section in ("results", "configs"):
        keys = list(aggregated.get(section, {}))
        if len(keys) == 1:
            return keys[0]

    raise ValueError("could not infer task name from aggregated results; pass --task explicitly")


def infer_repeats(aggregated: dict, task_name: str) -> int:
    task_config = aggregated.get("configs", {}).get(task_name, {})
    repeats = task_config.get("repeats")
    if repeats is not None:
        return int(repeats)
    task_settings = load_task_settings(find_task_config_path(task_name))
    return task_settings.repeats


def load_samples(run_dir: Path, task_name: str) -> list[dict]:
    sample_path = latest_file(run_dir, f"samples_{task_name}_*.jsonl")
    return [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]


def summarize_run(
    samples: list[dict],
    k: int,
    expected_n: int,
    *,
    reasoning_tags: list[tuple[str, str]] | None = None,
) -> dict[str, float]:
    samples_by_doc_id = {sample["doc_id"]: sample for sample in samples}
    if not samples_by_doc_id:
        raise ValueError("no valid samples found in run")

    metric_values = {"pass": [], "avg": [], "maj": []}
    for sample in samples_by_doc_id.values():
        completions = [str(response) for response in sample["resps"][0]]
        actual_n = len(completions)
        if actual_n != expected_n:
            raise ValueError(
                f"expected {expected_n} completions per problem, found {actual_n} for doc_id={sample['doc_id']}"
            )

        target = str(sample["target"])
        metric_values["pass"].append(
            score_pass_at_k(target, completions, expected_n, k, reasoning_tags)
        )
        metric_values["avg"].append(score_avg_at_n(target, completions, expected_n, reasoning_tags))
        metric_values["maj"].append(score_maj_at_n(target, completions, expected_n, reasoning_tags))

    num_docs = len(samples_by_doc_id)
    return {
        "num_docs": float(num_docs),
        "pass": sum(metric_values["pass"]) / num_docs,
        "avg": sum(metric_values["avg"]) / num_docs,
        "maj": sum(metric_values["maj"]) / num_docs,
        "pass_stddev": sample_stddev(metric_values["pass"]),
        "avg_stddev": sample_stddev(metric_values["avg"]),
        "maj_stddev": sample_stddev(metric_values["maj"]),
    }


def build_postprocess_payload(
    run_dir: Path,
    aggregated: dict,
    summary: dict[str, float],
    task_name: str,
    k: int,
    repeats: int,
) -> dict:
    return {
        "run_dir": str(run_dir),
        "model": resolve_model_identity(aggregated, run_dir),
        "task": task_name,
        "repeats": repeats,
        "k": k,
        "stddev_kind": "sample",
        "metrics": {
            f"pass@{k}": summary["pass"],
            f"avg@{repeats}": summary["avg"],
            f"maj@{repeats}": summary["maj"],
            "num_docs": int(summary["num_docs"]),
        },
        "stddev": {
            f"pass@{k}": summary["pass_stddev"],
            f"avg@{repeats}": summary["avg_stddev"],
            f"maj@{repeats}": summary["maj_stddev"],
        },
    }


def format_summary(
    run_dir: Path,
    aggregated: dict,
    summary: dict[str, float],
    task_name: str,
    k: int,
    n: int,
) -> str:
    model_name = resolve_model_identity(aggregated, run_dir)
    lines = [
        f"run_dir: {run_dir}",
        f"model: {model_name}",
        f"task: {task_name}",
        f"repeats: {n}",
        f"pass@{k}: {summary['pass']:.6f}",
        f"pass@{k} stddev: {summary['pass_stddev']:.6f}",
        f"avg@{n}: {summary['avg']:.6f}",
        f"avg@{n} stddev: {summary['avg_stddev']:.6f}",
        f"maj@{n}: {summary['maj']:.6f}",
        f"maj@{n} stddev: {summary['maj_stddev']:.6f}",
        f"docs: {int(summary['num_docs'])}",
        "stddev: sample",
    ]

    if k == 1:
        lines.append(
            f"note: pass@1 uses the Chen et al. estimator and matches avg@{n} by definition."
        )

    return "\n".join(lines)


def extract_suffix(path: Path, prefix: str) -> str:
    stem = path.stem
    if not stem.startswith(prefix):
        raise ValueError(f"{path.name} does not start with {prefix}")
    return stem[len(prefix) :]


def write_postprocess_artifacts(
    *,
    run_dir: Path,
    task_name: str | None = None,
    k: int = 1,
) -> tuple[Path, Path]:
    aggregated_path = latest_file(run_dir, "results_*.json")
    aggregated = json.loads(aggregated_path.read_text(encoding="utf-8"))
    resolved_task_name = task_name or infer_task_name(aggregated)
    repeats = infer_repeats(aggregated, resolved_task_name)
    samples = load_samples(run_dir, resolved_task_name)
    reasoning_tags = resolve_reasoning_tags(aggregated)
    summary = summarize_run(samples, k=k, expected_n=repeats, reasoning_tags=reasoning_tags)
    payload = build_postprocess_payload(
        run_dir,
        aggregated,
        summary,
        resolved_task_name,
        k,
        repeats,
    )
    rendered = format_summary(
        run_dir,
        aggregated,
        summary,
        resolved_task_name,
        k,
        repeats,
    )
    suffix = extract_suffix(aggregated_path, "results_")
    postprocess_path = run_dir / f"postprocess_{suffix}.json"
    summary_path = run_dir / f"summary_{suffix}.txt"
    postprocess_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(rendered + "\n", encoding="utf-8")
    return postprocess_path, summary_path


def main() -> None:
    args = parse_args()
    _, summary_path = write_postprocess_artifacts(
        run_dir=args.run_dir,
        task_name=args.task,
        k=args.k,
    )
    print(summary_path.read_text(encoding="utf-8").rstrip())


if __name__ == "__main__":
    main()
