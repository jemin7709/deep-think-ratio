from __future__ import annotations

import argparse
import json
from collections import Counter
from math import prod
from pathlib import Path

from scripts.common import find_task_config_path, load_task_settings

from .utils import extract_answer, is_equiv


TASK_NAME = "aime24_custom"


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


def canonical_answers(completions: list[str]) -> list[str]:
    return [extract_answer(completion) for completion in completions]


def majority_answer(answers: list[str]) -> str:
    return Counter(answers).most_common(1)[0][0]


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"pass@{k} requires k <= n, got n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    product = prod(1.0 - (k / denominator) for denominator in range(n - c + 1, n + 1))
    return 1.0 - product


def summarize_run(samples: list[dict], k: int, expected_n: int) -> dict[str, float]:
    samples_by_doc_id = {sample["doc_id"]: sample for sample in samples}
    if not samples_by_doc_id:
        raise ValueError("no valid samples found in run")

    totals = {"pass": 0.0, "avg": 0.0, "first": 0.0, "maj": 0.0}
    for sample in samples_by_doc_id.values():
        # Read raw responses because the default lm-eval filter keeps only the first repeat.
        completions = [str(response) for response in sample["resps"][0]]
        actual_n = len(completions)
        if actual_n != expected_n:
            raise ValueError(
                f"expected {expected_n} completions per problem, found {actual_n} for doc_id={sample['doc_id']}"
            )

        target = str(sample["target"])
        answers = canonical_answers(completions)
        scores = [int(is_equiv(answer, target)) for answer in answers]
        correct = sum(scores)

        totals["pass"] += pass_at_k(actual_n, correct, k)
        totals["avg"] += correct / actual_n
        totals["first"] += float(scores[0])
        totals["maj"] += float(is_equiv(majority_answer(answers), target))

    num_docs = len(samples_by_doc_id)

    return {
        "num_docs": float(num_docs),
        "pass": totals["pass"] / num_docs,
        "avg": totals["avg"] / num_docs,
        "first": totals["first"] / num_docs,
        "maj": totals["maj"] / num_docs,
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
        "model": aggregated.get("config", {}).get("model", "unknown"),
        "task": task_name,
        "repeats": repeats,
        "k": k,
        "metrics": {
            f"pass@{k}": summary["pass"],
            f"avg@{repeats}": summary["avg"],
            f"maj@{repeats}": summary["maj"],
            "first@1": summary["first"],
            "num_docs": int(summary["num_docs"]),
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
    model_name = aggregated.get("config", {}).get("model", "unknown")
    lines = [
        f"run_dir: {run_dir}",
        f"model: {model_name}",
        f"task: {task_name}",
        f"repeats: {n}",
        f"pass@{k}: {summary['pass']:.6f}",
        f"avg@{n}: {summary['avg']:.6f}",
        f"maj@{n}: {summary['maj']:.6f}",
        f"first@1: {summary['first']:.6f}",
        f"docs: {int(summary['num_docs'])}",
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
    summary = summarize_run(samples, k=k, expected_n=repeats)
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
