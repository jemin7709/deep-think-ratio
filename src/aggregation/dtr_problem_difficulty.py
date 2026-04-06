"""run 디렉터리에서 문항 난이도 지표와 bucket 요약을 계산한다."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from src.dtr.jsd_utils import (
    infer_task_name,
    dtr_results_path,
    latest_matching_file,
    load_aggregated_results,
)
from src.plot.dtr_problem_difficulty import plot_bucket_summary_to_png, plot_scatter_to_png
from tasks.aime24.utils import resolve_model_identity, resolve_reasoning_tags, score_match


DEFAULT_OUTPUT_DIR_NAME = "dtr_problem_difficulty"
OUTPUT_PLOT_FILENAME = "dtr_problem_difficulty_scatter.png"
OUTPUT_BUCKET_PLOT_FILENAME = "dtr_problem_difficulty_bucket.png"
OUTPUT_JSON_FILENAME = "dtr_problem_difficulty.json"


@dataclass(frozen=True)
class DtrSequenceRecord:
    """doc_id/repeat 단위의 DTR 결과."""

    dtr: float
    response_length: int


@dataclass(frozen=True)
class ProblemSequenceResult:
    """문항-반복 조합의 정답/난이도 성분."""

    doc_id: int
    repeat_index: int
    dtr: float
    response_length: int
    correct: bool


@dataclass(frozen=True)
class ProblemDifficultyRow:
    """문항별 집계 난이도 row."""

    doc_id: int
    accuracy: float
    difficulty_score: float
    mean_dtr: float
    mean_response_length: float
    difficulty_bucket: str
    correct_count: int
    repeat_count: int


@dataclass(frozen=True)
class BucketSummary:
    """난이도 bucket별 요약."""

    bucket: str
    num_problems: int
    mean_accuracy: float
    mean_dtr: float
    mean_response_length: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate per-problem difficulty from run-level DTR/repeat data."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dtr-path", type=Path)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--samples-path", type=Path)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--output-bucket-plot", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--title")
    return parser.parse_args()


def resolve_output_plot_path(
    output_plot: Path,
    default_name: str,
) -> Path:
    candidate = output_plot.resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / default_name
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")


def default_output_dir(run_dir: Path) -> Path:
    """문항 난이도 산출물의 기본 경로."""
    return run_dir / DEFAULT_OUTPUT_DIR_NAME


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None, Path, Path, Path]:
    run_dir = args.run_dir.resolve()
    dtr_path = (
        args.dtr_path.resolve() if args.dtr_path is not None else dtr_results_path(run_dir)
    )
    results_path = (
        args.results_path.resolve()
        if args.results_path is not None
        else latest_matching_file(run_dir, "results_*.json")
    )
    if args.samples_path is not None:
        samples_path = args.samples_path.resolve()
    else:
        samples_path = None

    output_dir = default_output_dir(run_dir)
    output_plot = (
        resolve_output_plot_path(args.output_plot, OUTPUT_PLOT_FILENAME)
        if args.output_plot is not None
        else output_dir / OUTPUT_PLOT_FILENAME
    )
    output_bucket_plot = (
        resolve_output_plot_path(
            args.output_bucket_plot,
            OUTPUT_BUCKET_PLOT_FILENAME,
        )
        if args.output_bucket_plot is not None
        else output_dir / OUTPUT_BUCKET_PLOT_FILENAME
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else output_dir / OUTPUT_JSON_FILENAME
    )

    return dtr_path, results_path, samples_path, output_plot, output_bucket_plot, output_json


def _results_suffix(results_path: Path) -> str | None:
    stem = results_path.stem
    if not stem.startswith("results_"):
        return None
    return stem[len("results_") :]


def resolve_samples_path(
    *,
    run_dir: Path,
    task_name: str,
    results_path: Path,
    samples_path: Path | None,
) -> Path:
    if samples_path is not None:
        return samples_path.resolve()

    suffix = _results_suffix(results_path)
    if suffix is not None:
        expected = run_dir / f"samples_{task_name}_{suffix}.jsonl"
        if expected.is_file():
            return expected

    candidates = sorted(run_dir.glob(f"samples_{task_name}_*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no samples_{task_name}_*.jsonl found under {run_dir}")
    raise FileNotFoundError(
        "multiple sample files found without an exact match for "
        f"{results_path.name}; pass --samples-path explicitly"
    )


def difficulty_bucket(accuracy: float) -> str:
    """정확도 임계값으로 difficulty bucket을 정한다."""
    if accuracy >= 0.75:
        return "easy"
    if accuracy <= 0.25:
        return "hard"
    return "medium"


def load_dtr_records_by_key(path: Path) -> dict[tuple[int, int], DtrSequenceRecord]:
    """doc/repeat 키로 DTR를 매핑한다."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    dtr_by_key: dict[tuple[int, int], DtrSequenceRecord] = {}
    for row in rows:
        if "num_tokens" not in row:
            raise ValueError(
                "DTR entry is missing num_tokens; regenerate DTR results with the current pipeline"
            )
        dtr_by_key[(int(row["doc_id"]), int(row["repeat_index"]))] = (
            DtrSequenceRecord(
                dtr=float(row["dtr"]),
                response_length=int(row["num_tokens"]),
            )
        )
    return dtr_by_key


def _score_response(
    target: str,
    response: str,
    reasoning_tags: list[tuple[str, str]] | None,
) -> bool:
    return score_match(target, response, reasoning_tags=reasoning_tags) == 1.0


def load_problem_rows(
    dtr_by_key: Mapping[tuple[int, int], tuple[float, int] | DtrSequenceRecord],
    samples_path: Path,
    reasoning_tags: list[tuple[str, str]] | None,
) -> list[ProblemSequenceResult]:
    """샘플에서 doc/repeat 반복 정답을 복원한다."""
    rows: list[ProblemSequenceResult] = []
    seen_keys: set[tuple[int, int]] = set()

    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            doc_id = int(sample["doc_id"])
            target = str(sample["target"])
            responses = [str(response) for response in sample["resps"][0]]
            for repeat_index, response in enumerate(responses):
                key = (doc_id, repeat_index)
                if key not in dtr_by_key:
                    raise ValueError(
                        f"missing DTR entry for doc_id={doc_id}, repeat={repeat_index}"
                    )
                record = dtr_by_key[key]
                if isinstance(record, tuple):
                    dtr_value, response_len = record
                else:
                    dtr_value, response_len = record.dtr, record.response_length
                rows.append(
                    ProblemSequenceResult(
                        doc_id=doc_id,
                        repeat_index=repeat_index,
                        dtr=float(dtr_value),
                        response_length=int(response_len),
                        correct=_score_response(target, response, reasoning_tags),
                    )
                )
                seen_keys.add(key)

    missing_from_samples = sorted(set(dtr_by_key) - seen_keys)
    if missing_from_samples:
        raise ValueError(
            "DTR entries without matching sample rows: "
            + ", ".join(f"{doc}:{repeat}" for doc, repeat in missing_from_samples[:5])
        )

    rows.sort(key=lambda row: (row.doc_id, row.repeat_index))
    return rows


def build_problem_rows(
    sequence_results: list[ProblemSequenceResult],
) -> list[ProblemDifficultyRow]:
    """repeat 단위를 doc 단위로 묶어 난이도 row를 만든다."""
    grouped: dict[int, list[ProblemSequenceResult]] = defaultdict(list)
    for row in sequence_results:
        grouped[row.doc_id].append(row)

    problem_rows: list[ProblemDifficultyRow] = []
    for doc_id in sorted(grouped):
        repeats = grouped[doc_id]
        repeat_count = len(repeats)
        correct_count = sum(1 for row in repeats if row.correct)
        accuracy = correct_count / repeat_count
        problem_rows.append(
            ProblemDifficultyRow(
                doc_id=doc_id,
                accuracy=accuracy,
                difficulty_score=1.0 - accuracy,
                mean_dtr=fmean(row.dtr for row in repeats),
                mean_response_length=fmean(row.response_length for row in repeats),
                difficulty_bucket=difficulty_bucket(accuracy),
                correct_count=correct_count,
                repeat_count=repeat_count,
            )
        )

    return problem_rows


def _ranks(values: list[float]) -> list[float]:
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks: list[float] = [0.0] * len(values)

    idx = 0
    while idx < len(values):
        start = idx
        current = sorted_pairs[idx][0]
        while idx + 1 < len(values) and sorted_pairs[idx + 1][0] == current:
            idx += 1
        end = idx
        avg_rank = (start + end + 2) / 2
        for _, original_index in sorted_pairs[start : end + 1]:
            ranks[original_index] = avg_rank
        idx += 1

    return ranks


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        raise ValueError("pearson_r requires at least one point")
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return numerator / math.sqrt(denom_x * denom_y)


def spearman_r(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        raise ValueError("spearman_r requires at least one point")
    return pearson_r(_ranks(xs), _ranks(ys))


def build_bucket_summary(
    bucket: str,
    problem_rows: list[ProblemDifficultyRow],
) -> BucketSummary:
    rows = [row for row in problem_rows if row.difficulty_bucket == bucket]
    if not rows:
        return BucketSummary(
            bucket=bucket,
            num_problems=0,
            mean_accuracy=0.0,
            mean_dtr=0.0,
            mean_response_length=0.0,
        )
    return BucketSummary(
        bucket=bucket,
        num_problems=len(rows),
        mean_accuracy=fmean(row.accuracy for row in rows),
        mean_dtr=fmean(row.mean_dtr for row in rows),
        mean_response_length=fmean(row.mean_response_length for row in rows),
    )


def build_bucket_summaries(
    problem_rows: list[ProblemDifficultyRow],
) -> list[BucketSummary]:
    return [
        build_bucket_summary(bucket, problem_rows)
        for bucket in ("easy", "medium", "hard")
    ]


def build_title(
    run_dir: Path,
    task_name: str,
    model_name: str,
    user_title: str | None,
) -> str:
    if user_title is not None:
        return user_title
    return f"{run_dir.name} | {task_name} | {model_name} | Problem Difficulty"


def write_summary_json(
    *,
    run_dir: Path,
    task_name: str,
    model_name: str,
    dtr_path: Path,
    results_path: Path,
    samples_path: Path,
    output_path: Path,
    problems: list[ProblemDifficultyRow],
    bucket_summaries: list[BucketSummary],
    spearman: float,
) -> None:
    payload = {
        "run_dir": str(run_dir),
        "task": task_name,
        "model": model_name,
        "dtr_path": str(dtr_path),
        "results_path": str(results_path),
        "samples_path": str(samples_path),
        "num_problems": len(problems),
        "mean_accuracy": fmean(row.accuracy for row in problems) if problems else 0.0,
        "mean_dtr": fmean(row.mean_dtr for row in problems) if problems else 0.0,
        "mean_response_length": (
            fmean(row.mean_response_length for row in problems) if problems else 0.0
        ),
        "spearman_r": spearman,
        "bucket_summaries": [asdict(summary) for summary in bucket_summaries],
        "problems": [asdict(row) for row in problems],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_summary(num_problems: int, spearman: float) -> None:
    print(f"num_problems: {num_problems}")
    print(f"Spearman(difficulty_score, mean_dtr): {spearman:.6f}")


def main() -> None:
    args = parse_args()
    dtr_path, results_path, samples_path, output_plot, output_bucket_plot, output_json = (
        resolve_paths(args)
    )

    aggregated = (
        load_aggregated_results(results_path.parent)
        if args.results_path is None
        else json.loads(results_path.read_text(encoding="utf-8"))
    )
    task_name = infer_task_name(aggregated)
    model_name = resolve_model_identity(aggregated, results_path.parent)
    reasoning_tags = resolve_reasoning_tags(aggregated)
    samples_path = resolve_samples_path(
        run_dir=results_path.parent,
        task_name=task_name,
        results_path=results_path,
        samples_path=samples_path,
    )

    dtr_by_key = load_dtr_records_by_key(dtr_path)
    sequence_rows = load_problem_rows(dtr_by_key, samples_path, reasoning_tags)
    problems = build_problem_rows(sequence_rows)
    bucket_summaries = build_bucket_summaries(problems)
    spearman = spearman_r(
        [row.difficulty_score for row in problems],
        [row.mean_dtr for row in problems],
    )

    plot_scatter_to_png(
        points=problems,
        spearman=spearman,
        output_path=output_plot,
        title=build_title(results_path.parent, task_name, model_name, args.title),
    )
    plot_bucket_summary_to_png(
        bucket_summaries=bucket_summaries,
        output_path=output_bucket_plot,
        title=build_title(results_path.parent, task_name, model_name, args.title),
    )
    write_summary_json(
        run_dir=results_path.parent,
        task_name=task_name,
        model_name=model_name,
        dtr_path=dtr_path,
        results_path=results_path,
        samples_path=samples_path,
        output_path=output_json,
        problems=problems,
        bucket_summaries=bucket_summaries,
        spearman=spearman,
    )
    print_summary(len(problems), spearman)
    print(f"Saved plot: {output_plot}")
    print(f"Saved plot: {output_bucket_plot}")
    print(f"Saved summary: {output_json}")


if __name__ == "__main__":
    main()
