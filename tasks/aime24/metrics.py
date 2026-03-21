from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tasks.aime24.utils import extract_answer, is_equiv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pass@k, avg@n, and maj@n from an lm-eval AIME24 run."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task", default="aime24_sc_25")
    parser.add_argument("--k", default=1, type=int)
    parser.add_argument("--n", default=25, type=int)
    return parser.parse_args()


def latest_file(run_dir: Path, pattern: str) -> Path:
    matches = list(run_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no files matched {pattern} under {run_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_aggregated(run_dir: Path) -> dict:
    return json.loads(latest_file(run_dir, "results_*.json").read_text())


def load_samples(run_dir: Path, task_name: str) -> list[dict]:
    sample_path = latest_file(run_dir, f"samples_{task_name}_*.jsonl")
    return [json.loads(line) for line in sample_path.read_text().splitlines() if line]


def group_samples(samples: list[dict]) -> dict[int, dict]:
    grouped = {}
    for sample in samples:
        if sample["doc_id"] not in grouped:
            grouped[sample["doc_id"]] = sample
    return grouped


def raw_completions(sample: dict) -> list[str]:
    return [str(response) for response in sample["resps"][0]]


def canonical_answers(completions: list[str]) -> list[str]:
    return [extract_answer(completion) for completion in completions]


def score_answers(answers: list[str], target: str) -> list[int]:
    return [int(is_equiv(answer, target)) for answer in answers]


def majority_answer(answers: list[str]) -> str:
    counts = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts, key=counts.get)


def pass_at_k(n: int, c: int, k: int) -> float:
    if k > n:
        raise ValueError(f"pass@{k} requires k <= n, got n={n}")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    product = 1.0
    for denominator in range(n - c + 1, n + 1):
        product *= 1.0 - (k / denominator)
    return 1.0 - product


def summarize_problem(sample: dict, k: int, expected_n: int) -> dict[str, float]:
    completions = raw_completions(sample)
    actual_n = len(completions)
    if actual_n != expected_n:
        raise ValueError(
            f"expected {expected_n} completions per problem, found {actual_n} for doc_id={sample['doc_id']}"
        )

    target = str(sample["target"])
    answers = canonical_answers(completions)
    scores = score_answers(answers, target)
    correct = sum(scores)

    return {
        "pass": pass_at_k(actual_n, correct, k),
        "avg": correct / actual_n,
        "first": float(scores[0]),
        "maj": float(is_equiv(majority_answer(answers), target)),
    }


def summarize_run(samples: list[dict], k: int, expected_n: int) -> dict[str, float]:
    grouped = group_samples(samples)
    per_problem = [summarize_problem(sample, k, expected_n) for sample in grouped.values()]

    return {
        "num_docs": float(len(per_problem)),
        "pass": sum(problem["pass"] for problem in per_problem) / len(per_problem),
        "avg": sum(problem["avg"] for problem in per_problem) / len(per_problem),
        "first": sum(problem["first"] for problem in per_problem) / len(per_problem),
        "maj": sum(problem["maj"] for problem in per_problem) / len(per_problem),
    }


def format_summary(
    run_dir: Path,
    aggregated: dict,
    summary: dict[str, float],
    k: int,
    n: int,
) -> str:
    model_name = aggregated.get("config", {}).get("model", "unknown")
    lines = [
        f"run_dir: {run_dir}",
        f"model: {model_name}",
        "task: aime24_sc_25",
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


def main() -> None:
    args = parse_args()
    aggregated = load_aggregated(args.run_dir)
    samples = load_samples(args.run_dir, args.task)
    summary = summarize_run(samples, k=args.k, expected_n=args.n)
    print(format_summary(args.run_dir, aggregated, summary, args.k, args.n))


if __name__ == "__main__":
    main()
