"""여러 run의 DTR-length scatter summary를 합쳐 하나의 scatter plot을 만든다."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from src.aggregation.dtr_length_scatter import (
    OUTPUT_JSON_FILENAME,
    SequenceLengthPoint,
    pearson_r,
)
from src.plot.dtr_length_scatter import plot_to_png


DEFAULT_INPUT_ROOT = Path("results")
DEFAULT_AGGREGATE_DIR_NAME = "dtr_length_scatter_aggregated"
OUTPUT_PLOT_FILENAME = "dtr_length_scatter_all_runs.png"
OUTPUT_JSON_FILENAME_AGGREGATED = "dtr_length_scatter_all_runs.json"


@dataclass(frozen=True)
class SourceSummary:
    """개별 run에서 만든 scatter summary."""

    summary_path: Path
    run_dir: Path
    task: str
    model: str
    points: list[SequenceLengthPoint]


@dataclass(frozen=True)
class AggregatedPoint:
    """여러 run을 합친 뒤 plot에 쓰는 sequence-level 점."""

    run_dir: str
    doc_id: int
    repeat_index: int
    dtr: float
    response_length: int
    is_correct: bool | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate run-level dtr_length_scatter summaries into one plot."
    )
    parser.add_argument("input_root", nargs="?", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--aggregate-dir-name", default=DEFAULT_AGGREGATE_DIR_NAME)
    parser.add_argument("--title")
    return parser.parse_args()


def discover_summary_paths(input_root: Path, aggregate_dir_name: str) -> list[Path]:
    excluded_dir = input_root / aggregate_dir_name
    paths = [
        path
        for path in sorted(input_root.rglob(OUTPUT_JSON_FILENAME))
        if excluded_dir not in path.parents
    ]
    if not paths:
        raise FileNotFoundError(f"no {OUTPUT_JSON_FILENAME} found under {input_root}")
    return paths


def load_source_summary(summary_path: Path) -> SourceSummary:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    points = [
        SequenceLengthPoint(
            doc_id=int(entry["doc_id"]),
            repeat_index=int(entry["repeat_index"]),
            dtr=float(entry["dtr"]),
            response_length=int(entry["response_length"]),
            is_correct=(
                None
                if entry.get("is_correct") is None
                else bool(entry["is_correct"])
            ),
        )
        for entry in payload["points"]
    ]
    return SourceSummary(
        summary_path=summary_path,
        run_dir=Path(payload["run_dir"]),
        task=str(payload["task"]),
        model=str(payload["model"]),
        points=points,
    )


def load_source_summaries(summary_paths: list[Path]) -> list[SourceSummary]:
    summaries = [load_source_summary(path) for path in summary_paths]
    if not summaries:
        raise ValueError("need at least one dtr_length_scatter summary")

    first = summaries[0]
    for summary in summaries[1:]:
        if summary.task != first.task or summary.model != first.model:
            raise ValueError("all source summaries must share task and model")
    return summaries


def aggregate_points(summaries: list[SourceSummary]) -> list[AggregatedPoint]:
    points: list[AggregatedPoint] = []
    for summary in summaries:
        for point in summary.points:
            points.append(
                AggregatedPoint(
                    run_dir=str(summary.run_dir),
                    doc_id=point.doc_id,
                    repeat_index=point.repeat_index,
                    dtr=point.dtr,
                    response_length=point.response_length,
                    is_correct=point.is_correct,
                )
            )
    points.sort(key=lambda point: (point.dtr, point.run_dir, point.doc_id, point.repeat_index))
    return points


def build_title(
    summaries: list[SourceSummary],
    point_count: int,
    user_title: str | None,
) -> str:
    if user_title is not None:
        return user_title
    first = summaries[0]
    return (
        f"{first.task} | {first.model} | DTR vs Response Length "
        f"({len(summaries)} runs, {point_count} points)"
    )


def write_aggregated_json(
    *,
    input_root: Path,
    aggregate_dir: Path,
    summaries: list[SourceSummary],
    points: list[AggregatedPoint],
    plot_path: Path,
) -> Path:
    dtrs = [point.dtr for point in points]
    lengths = [point.response_length for point in points]
    first = summaries[0]
    output_path = aggregate_dir / OUTPUT_JSON_FILENAME_AGGREGATED
    payload = {
        "input_root": str(input_root),
        "aggregate_dir": str(aggregate_dir),
        "task": first.task,
        "model": first.model,
        "source_count": len(summaries),
        "source_paths": [str(summary.summary_path) for summary in summaries],
        "run_dirs": [str(summary.run_dir) for summary in summaries],
        "num_sequences": len(points),
        "pearson_r": pearson_r(dtrs, [float(length) for length in lengths]),
        "dtr_min": min(dtrs),
        "dtr_max": max(dtrs),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": fmean(lengths),
        "plot_path": str(plot_path),
        "points": [asdict(point) for point in points],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def print_summary(points: list[AggregatedPoint], aggregated_json: Path) -> None:
    dtrs = [point.dtr for point in points]
    lengths = [point.response_length for point in points]
    print(f"num_sequences: {len(points)}")
    print(f"dtr_range: [{min(dtrs):.4f}, {max(dtrs):.4f}]")
    print(f"length_range: [{min(lengths)}, {max(lengths)}]")
    print(f"length_mean: {fmean(lengths):.4f}")
    print(f"Pearson r: {pearson_r(dtrs, [float(length) for length in lengths]):.6f}")
    print(f"Saved aggregated summary: {aggregated_json}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    aggregate_dir = input_root / args.aggregate_dir_name

    summary_paths = discover_summary_paths(input_root, args.aggregate_dir_name)
    summaries = load_source_summaries(summary_paths)
    points = aggregate_points(summaries)
    if not points:
        raise ValueError("no sequence points found across source summaries")

    aggregate_dir.mkdir(parents=True, exist_ok=True)
    plot_path = aggregate_dir / OUTPUT_PLOT_FILENAME
    plot_to_png(
        points=points,
        pearson=pearson_r(
            [point.dtr for point in points],
            [float(point.response_length) for point in points],
        ),
        output_path=plot_path,
        title=build_title(summaries, len(points), args.title),
    )
    aggregated_json = write_aggregated_json(
        input_root=input_root,
        aggregate_dir=aggregate_dir,
        summaries=summaries,
        points=points,
        plot_path=plot_path,
    )
    print_summary(points, aggregated_json)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
