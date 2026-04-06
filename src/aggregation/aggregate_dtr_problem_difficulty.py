"""개별 run의 `dtr_problem_difficulty.json`을 읽어 난이도 통계를 통합한다."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, stdev

from src.aggregation.dtr_problem_difficulty import (
    BucketSummary,
    ProblemDifficultyRow,
    build_bucket_summaries,
    spearman_r,
)
from src.plot.dtr_problem_difficulty import plot_bucket_summary_to_png, plot_scatter_to_png


DEFAULT_INPUT_ROOT = Path("results")
DEFAULT_AGGREGATE_DIR_NAME = "dtr_problem_difficulty_aggregated"
OUTPUT_PLOT_FILENAME = "dtr_problem_difficulty_scatter_all_runs.png"
OUTPUT_BUCKET_PLOT_FILENAME = "dtr_problem_difficulty_bucket_all_runs.png"
OUTPUT_JSON_FILENAME_AGGREGATED = "dtr_problem_difficulty_all_runs.json"


@dataclass(frozen=True)
class SourceSummary:
    """run-level 난이도 요약 JSON을 파싱한 결과."""

    summary_path: Path
    run_dir: Path
    task: str
    model: str
    spearman_r: float
    num_problems: int
    mean_accuracy: float
    mean_dtr: float
    mean_response_length: float
    bucket_summaries: list[BucketSummary]
    problems: list[ProblemDifficultyRow]


@dataclass(frozen=True)
class AggregateBucketSummary:
    """bucket 단위 run-level 평균/표준편차."""

    bucket: str
    source_count: int
    num_problems_mean: float
    num_problems_std: float
    mean_accuracy_mean: float
    mean_accuracy_std: float
    mean_dtr_mean: float
    mean_dtr_std: float
    mean_response_length_mean: float
    mean_response_length_std: float

    @property
    def num_problems(self) -> float:
        return self.num_problems_mean

    @property
    def mean_accuracy(self) -> float:
        return self.mean_accuracy_mean

    @property
    def mean_dtr(self) -> float:
        return self.mean_dtr_mean

    @property
    def mean_response_length(self) -> float:
        return self.mean_response_length_mean


@dataclass(frozen=True)
class AggregatedSummary:
    """seed 수준 요약."""

    task: str
    model: str
    source_count: int
    source_paths: list[Path]
    run_dirs: list[Path]
    overall_num_problems: dict[str, float]
    overall_accuracy: dict[str, float]
    overall_difficulty_mean: dict[str, float]
    overall_dtr: dict[str, float]
    overall_response_length: dict[str, float]
    spearman: dict[str, float]
    bucket_summaries: dict[str, AggregateBucketSummary]
    scatter_points: list[ProblemDifficultyRow]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate run-level dtr_problem_difficulty summaries."
    )
    parser.add_argument("input_root", nargs="?", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--aggregate-dir-name", default=DEFAULT_AGGREGATE_DIR_NAME)
    parser.add_argument("--title")
    return parser.parse_args()


def discover_summary_paths(
    input_root: Path,
    aggregate_dir_name: str,
) -> list[Path]:
    excluded_dir = input_root / aggregate_dir_name
    paths = [
        path
        for path in sorted(input_root.rglob("dtr_problem_difficulty.json"))
        if excluded_dir not in path.parents
    ]
    if not paths:
        raise FileNotFoundError(
            "no dtr_problem_difficulty.json found under "
            f"{input_root} (excluding {aggregate_dir_name})"
        )
    return paths


def load_source_summary(path: Path) -> SourceSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems = [
        ProblemDifficultyRow(
            doc_id=int(entry["doc_id"]),
            accuracy=float(entry["accuracy"]),
            difficulty_score=float(entry["difficulty_score"]),
            mean_dtr=float(entry["mean_dtr"]),
            mean_response_length=float(entry["mean_response_length"]),
            difficulty_bucket=str(entry["difficulty_bucket"]),
            correct_count=int(entry["correct_count"]),
            repeat_count=int(entry["repeat_count"]),
        )
        for entry in payload.get("problems", [])
    ]
    bucket_payload = payload.get("bucket_summaries")
    if bucket_payload is None:
        bucket_summaries = build_bucket_summaries(problems)
    else:
        bucket_summaries = [
            BucketSummary(
                bucket=str(entry["bucket"]),
                num_problems=int(entry["num_problems"]),
                mean_accuracy=float(entry["mean_accuracy"]),
                mean_dtr=float(entry["mean_dtr"]),
                mean_response_length=float(entry["mean_response_length"]),
            )
            for entry in bucket_payload
        ]
    return SourceSummary(
        summary_path=path,
        run_dir=Path(payload["run_dir"]),
        task=str(payload["task"]),
        model=str(payload["model"]),
        spearman_r=(
            float(payload["spearman_r"])
            if payload.get("spearman_r") is not None
            else (
                spearman_r(
                    [problem.difficulty_score for problem in problems],
                    [problem.mean_dtr for problem in problems],
                )
                if problems
                else 0.0
            )
        ),
        num_problems=(
            int(payload["num_problems"])
            if payload.get("num_problems") is not None
            else len(problems)
        ),
        mean_accuracy=(
            float(payload["mean_accuracy"])
            if payload.get("mean_accuracy") is not None
            else (fmean(problem.accuracy for problem in problems) if problems else 0.0)
        ),
        mean_dtr=(
            float(payload["mean_dtr"])
            if payload.get("mean_dtr") is not None
            else (fmean(problem.mean_dtr for problem in problems) if problems else 0.0)
        ),
        mean_response_length=(
            float(payload["mean_response_length"])
            if payload.get("mean_response_length") is not None
            else (
                fmean(problem.mean_response_length for problem in problems)
                if problems
                else 0.0
            )
        ),
        bucket_summaries=bucket_summaries,
        problems=problems,
    )


def load_source_summaries(summary_paths: list[Path]) -> list[SourceSummary]:
    summaries = [load_source_summary(path) for path in summary_paths]
    if not summaries:
        raise ValueError("need at least one dtr_problem_difficulty summary")

    first = summaries[0]
    for summary in summaries[1:]:
        if summary.task != first.task or summary.model != first.model:
            raise ValueError("all source summaries must share task and model")

    return summaries


def sample_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def aggregate_bucket_summaries(
    summaries: list[SourceSummary],
) -> dict[str, AggregateBucketSummary]:
    if not summaries:
        raise ValueError("need at least one source summary")

    bucket_keys = ["easy", "medium", "hard"]
    extra_keys = sorted(
        {
            entry.bucket
            for summary in summaries
            for entry in summary.bucket_summaries
            if entry.bucket not in {"easy", "medium", "hard"}
        }
    )
    bucket_keys.extend(extra_keys)

    aggregated: dict[str, AggregateBucketSummary] = {}
    for bucket in bucket_keys:
        bucket_entries = [
            entry
            for summary in summaries
            for entry in summary.bucket_summaries
            if entry.bucket == bucket
        ]
        if bucket_entries:
            num_problems_values = [entry.num_problems for entry in bucket_entries]
            mean_accuracy_values = [entry.mean_accuracy for entry in bucket_entries]
            mean_dtr_values = [entry.mean_dtr for entry in bucket_entries]
            mean_response_values = [entry.mean_response_length for entry in bucket_entries]
            num_source = len(bucket_entries)
            aggregated[bucket] = AggregateBucketSummary(
                bucket=bucket,
                source_count=num_source,
                num_problems_mean=fmean(num_problems_values),
                num_problems_std=sample_std([float(value) for value in num_problems_values]),
                mean_accuracy_mean=fmean(mean_accuracy_values),
                mean_accuracy_std=sample_std(mean_accuracy_values),
                mean_dtr_mean=fmean(mean_dtr_values),
                mean_dtr_std=sample_std(mean_dtr_values),
                mean_response_length_mean=fmean(mean_response_values),
                mean_response_length_std=sample_std(mean_response_values),
            )
        else:
            aggregated[bucket] = AggregateBucketSummary(
                bucket=bucket,
                source_count=0,
                num_problems_mean=0.0,
                num_problems_std=0.0,
                mean_accuracy_mean=0.0,
                mean_accuracy_std=0.0,
                mean_dtr_mean=0.0,
                mean_dtr_std=0.0,
                mean_response_length_mean=0.0,
                mean_response_length_std=0.0,
            )

    return aggregated


def run_level_summary_payload(
    summaries: list[SourceSummary],
) -> AggregatedSummary:
    overall = AggregatedSummary(
        task=summaries[0].task,
        model=summaries[0].model,
        source_count=len(summaries),
        source_paths=[summary.summary_path for summary in summaries],
        run_dirs=[summary.run_dir for summary in summaries],
        overall_num_problems={
            "mean": fmean(summary.num_problems for summary in summaries),
            "std": sample_std([float(summary.num_problems) for summary in summaries]),
        },
        overall_accuracy={
            "mean": fmean(summary.mean_accuracy for summary in summaries),
            "std": sample_std([summary.mean_accuracy for summary in summaries]),
        },
        overall_difficulty_mean={
            "mean": fmean(1.0 - summary.mean_accuracy for summary in summaries),
            "std": sample_std([1.0 - summary.mean_accuracy for summary in summaries]),
        },
        overall_dtr={
            "mean": fmean(summary.mean_dtr for summary in summaries),
            "std": sample_std([summary.mean_dtr for summary in summaries]),
        },
        overall_response_length={
            "mean": fmean(summary.mean_response_length for summary in summaries),
            "std": sample_std([summary.mean_response_length for summary in summaries]),
        },
        spearman={
            "mean": fmean(summary.spearman_r for summary in summaries),
            "std": sample_std([summary.spearman_r for summary in summaries]),
        },
        bucket_summaries=aggregate_bucket_summaries(summaries),
        scatter_points=[
            problem
            for summary in summaries
            for problem in summary.problems
        ],
    )
    return overall


def build_title(
    summaries: list[SourceSummary],
    num_points: int | float,
    user_title: str | None,
) -> str:
    if user_title is not None:
        return user_title
    first = summaries[0]
    return (
        f"{first.task} | {first.model} | Problem Difficulty "
        f"({len(summaries)} runs, {num_points} problems)"
    )


def write_aggregated_json(
    *,
    input_root: Path,
    aggregate_dir: Path,
    summaries: list[SourceSummary],
    run_summary: AggregatedSummary,
    output_scatter_path: Path,
    output_bucket_plot_path: Path,
) -> Path:
    output_path = aggregate_dir / OUTPUT_JSON_FILENAME_AGGREGATED
    payload = {
        "input_root": str(input_root),
        "aggregate_dir": str(aggregate_dir),
        "task": run_summary.task,
        "model": run_summary.model,
        "source_count": run_summary.source_count,
        "source_paths": [str(path) for path in run_summary.source_paths],
        "run_dirs": [str(run_dir) for run_dir in run_summary.run_dirs],
        "spearman_r_mean": run_summary.spearman["mean"],
        "spearman_r_std": run_summary.spearman["std"],
        "num_problems_mean": run_summary.overall_num_problems["mean"],
        "num_problems_std": run_summary.overall_num_problems["std"],
        "mean_accuracy_mean": run_summary.overall_accuracy["mean"],
        "mean_accuracy_std": run_summary.overall_accuracy["std"],
        "mean_difficulty_score_mean": run_summary.overall_difficulty_mean["mean"],
        "mean_difficulty_score_std": run_summary.overall_difficulty_mean["std"],
        "mean_dtr_mean": run_summary.overall_dtr["mean"],
        "mean_dtr_std": run_summary.overall_dtr["std"],
        "mean_response_length_mean": run_summary.overall_response_length["mean"],
        "mean_response_length_std": run_summary.overall_response_length["std"],
        "bucket_summaries": [asdict(entry) for entry in run_summary.bucket_summaries.values()],
        "scatter_points": [asdict(point) for point in run_summary.scatter_points],
        "plot_path": str(output_scatter_path),
        "bucket_plot_path": str(output_bucket_plot_path),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    aggregate_dir = input_root / args.aggregate_dir_name

    summary_paths = discover_summary_paths(input_root, args.aggregate_dir_name)
    summaries = load_source_summaries(summary_paths)
    run_summary = run_level_summary_payload(summaries)

    aggregate_dir.mkdir(parents=True, exist_ok=True)
    output_scatter = aggregate_dir / OUTPUT_PLOT_FILENAME
    output_bucket_plot = aggregate_dir / OUTPUT_BUCKET_PLOT_FILENAME
    title_points = len(run_summary.scatter_points)
    plot_scatter_to_png(
        points=run_summary.scatter_points,
        spearman=run_summary.spearman["mean"],
        output_path=output_scatter,
        title=build_title(summaries, title_points, args.title),
    )
    plot_bucket_summary_to_png(
        bucket_summaries=[entry for entry in run_summary.bucket_summaries.values()],
        output_path=output_bucket_plot,
        title=build_title(summaries, title_points, args.title),
    )
    output_json = write_aggregated_json(
        input_root=input_root,
        aggregate_dir=aggregate_dir,
        summaries=summaries,
        run_summary=run_summary,
        output_scatter_path=output_scatter,
        output_bucket_plot_path=output_bucket_plot,
    )

    print(f"Loaded {len(summaries)} source summaries from {len(summary_paths)} files")
    print(f"Saved aggregated summary: {output_json}")
    print(f"Saved plot: {output_scatter}")
    print(f"Saved plot: {output_bucket_plot}")


if __name__ == "__main__":
    main()
