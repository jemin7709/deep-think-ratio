"""여러 run의 DTR-length scatter summary를 합쳐 하나의 scatter plot을 만든다."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from src.aggregation.dtr_length_scatter import (
    DEFAULT_LENGTH_MODE,
    REASONING_LENGTH_MODE,
    SequenceLengthPoint,
    build_length_axis_label,
    build_mode_filename,
    build_title as build_run_title,
    pearson_r,
    validate_length_mode,
)
from src.plot.dtr_length_scatter import plot_to_png


DEFAULT_INPUT_ROOT = Path("results")
DEFAULT_AGGREGATE_DIR_NAME = "dtr_length_scatter_aggregated"
OUTPUT_PLOT_FILENAME = "dtr_length_scatter_all_runs.png"
OUTPUT_JSON_FILENAME_AGGREGATED = "dtr_length_scatter_all_runs.json"
OUTPUT_AGGREGATED_STEM = "dtr_length_scatter_all_runs"


@dataclass(frozen=True)
class SourceSummary:
    """개별 run에서 만든 scatter summary."""

    summary_path: Path
    run_dir: Path
    task: str
    model: str
    length_mode: str
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
    parser.add_argument(
        "--length-mode",
        choices=(DEFAULT_LENGTH_MODE, REASONING_LENGTH_MODE),
        default=DEFAULT_LENGTH_MODE,
    )
    parser.add_argument("--title")
    return parser.parse_args()


def build_aggregated_plot_filename(length_mode: str) -> str:
    return build_mode_filename(OUTPUT_AGGREGATED_STEM, length_mode, ".png")


def build_aggregated_summary_filename(length_mode: str) -> str:
    return build_mode_filename(OUTPUT_AGGREGATED_STEM, length_mode, ".json")


def discover_summary_paths(
    input_root: Path,
    aggregate_dir_name: str,
    length_mode: str,
) -> list[Path]:
    filename = build_mode_filename("dtr_length_scatter", validate_length_mode(length_mode), ".json")
    excluded_dir = input_root / aggregate_dir_name
    paths = [
        path
        for path in sorted(input_root.rglob(filename))
        if excluded_dir not in path.parents
    ]
    if not paths:
        raise FileNotFoundError(f"no {filename} found under {input_root}")
    return paths


def load_source_summary(
    summary_path: Path,
    *,
    length_mode: str = DEFAULT_LENGTH_MODE,
) -> SourceSummary:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload_length_mode = str(payload.get("length_mode", ""))
    if payload_length_mode != validate_length_mode(length_mode):
        raise ValueError(
            f"summary {summary_path} has length_mode={payload_length_mode!r}, expected {length_mode!r}"
        )
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
        length_mode=payload_length_mode,
        points=points,
    )


def load_source_summaries(
    summary_paths: list[Path],
    *,
    length_mode: str = DEFAULT_LENGTH_MODE,
) -> list[SourceSummary]:
    summaries = [load_source_summary(path, length_mode=length_mode) for path in summary_paths]
    if not summaries:
        raise ValueError("need at least one dtr_length_scatter summary")

    first = summaries[0]
    for summary in summaries[1:]:
        if summary.task != first.task or summary.model != first.model:
            raise ValueError("all source summaries must share task and model")
        if summary.length_mode != first.length_mode:
            raise ValueError("all source summaries must share length_mode")
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


def _correctness_length_stats(
    points: list[AggregatedPoint],
) -> dict[str, int | float | None]:
    correct_lengths = [
        point.response_length for point in points if point.is_correct is True
    ]
    incorrect_lengths = [
        point.response_length for point in points if point.is_correct is False
    ]

    return {
        "correct_count": len(correct_lengths),
        "incorrect_count": len(incorrect_lengths),
        "correct_length_mean": fmean(correct_lengths) if correct_lengths else None,
        "incorrect_length_mean": fmean(incorrect_lengths)
        if incorrect_lengths
        else None,
    }


def build_title(
    summaries: list[SourceSummary],
    point_count: int,
    user_title: str | None,
    length_mode: str = DEFAULT_LENGTH_MODE,
) -> str:
    if user_title is not None:
        return user_title
    first = summaries[0]
    return (
        f"{first.task} | {first.model} | "
        f"{build_run_title(first.run_dir, first.task, first.model, None, length_mode).split(' | ', 3)[-1]} "
        f"({len(summaries)} runs, {point_count} points)"
    )


def write_aggregated_json(
    *,
    input_root: Path,
    aggregate_dir: Path,
    summaries: list[SourceSummary],
    points: list[AggregatedPoint],
    plot_path: Path,
    length_mode: str,
) -> Path:
    dtrs = [point.dtr for point in points]
    lengths = [point.response_length for point in points]
    first = summaries[0]
    output_path = aggregate_dir / build_aggregated_summary_filename(length_mode)
    payload = {
        "input_root": str(input_root),
        "aggregate_dir": str(aggregate_dir),
        "task": first.task,
        "model": first.model,
        "length_mode": validate_length_mode(length_mode),
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
        **_correctness_length_stats(points),
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
    length_mode = validate_length_mode(args.length_mode)

    summary_paths = discover_summary_paths(
        input_root,
        args.aggregate_dir_name,
        length_mode,
    )
    summaries = load_source_summaries(summary_paths, length_mode=length_mode)
    points = aggregate_points(summaries)
    if not points:
        raise ValueError("no sequence points found across source summaries")

    aggregate_dir.mkdir(parents=True, exist_ok=True)
    plot_path = aggregate_dir / build_aggregated_plot_filename(length_mode)
    plot_to_png(
        points=points,
        pearson=pearson_r(
            [point.dtr for point in points],
            [float(point.response_length) for point in points],
        ),
        output_path=plot_path,
        title=build_title(summaries, len(points), args.title, length_mode=length_mode),
        y_label=build_length_axis_label(length_mode),
    )
    aggregated_json = write_aggregated_json(
        input_root=input_root,
        aggregate_dir=aggregate_dir,
        summaries=summaries,
        points=points,
        plot_path=plot_path,
        length_mode=length_mode,
    )
    print_summary(points, aggregated_json)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
