"""run 디렉터리의 DTR와 응답 길이를 묶어 scatter plot 산출물을 만든다."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from src.dtr.jsd_utils import infer_task_name
from src.dtr.jsd_utils import (
    dtr_results_path,
    latest_matching_file,
    load_aggregated_results,
)
from src.plot.dtr_length_scatter import plot_to_png


DEFAULT_OUTPUT_DIR_NAME = "dtr_length_scatter"
OUTPUT_PLOT_FILENAME = "dtr_length_scatter.png"
OUTPUT_JSON_FILENAME = "dtr_length_scatter.json"


@dataclass(frozen=True)
class SequenceLengthPoint:
    """Per-repeat 수준에서 DTR와 응답 길이를 합친 점."""

    doc_id: int
    repeat_index: int
    dtr: float
    response_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sequence-level DTR vs response length scatter."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dtr-path", type=Path)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--title")
    return parser.parse_args()


def resolve_output_plot_path(output_plot: Path | None) -> Path | None:
    if output_plot is None:
        return None

    candidate = output_plot.resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / OUTPUT_PLOT_FILENAME
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")


def default_output_dir(run_dir: Path) -> Path:
    """개별 run의 DTR-length scatter 산출물을 모으는 기본 디렉터리."""
    return run_dir / DEFAULT_OUTPUT_DIR_NAME


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None, Path]:
    run_dir = args.run_dir.resolve()
    dtr_path = (
        args.dtr_path.resolve() if args.dtr_path is not None else dtr_results_path(run_dir)
    )
    results_path = (
        args.results_path.resolve()
        if args.results_path is not None
        else latest_matching_file(run_dir, "results_*.json")
    )
    output_dir = default_output_dir(run_dir)
    output_plot = (
        resolve_output_plot_path(args.output_plot)
        if args.output_plot is not None
        else output_dir / OUTPUT_PLOT_FILENAME
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else output_dir / OUTPUT_JSON_FILENAME
    )
    return dtr_path, results_path, output_plot, output_json


def resolve_model_name(aggregated: dict[str, Any], run_dir: Path) -> str:
    """results payload 안에서 사람이 읽을 모델 식별자를 고른다."""
    config = aggregated.get("config", {})
    metadata = config.get("metadata", {})
    model_args = config.get("model_args", {})
    metadata_model_args = metadata.get("model_args", {})
    candidates = [
        model_args.get("pretrained"),
        metadata_model_args.get("pretrained"),
        config.get("model_name"),
        metadata.get("model_name"),
        run_dir.parents[1].name if len(run_dir.parents) > 1 else None,
        config.get("model"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown"


def load_points(path: Path) -> list[SequenceLengthPoint]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    points: list[SequenceLengthPoint] = []

    for row in rows:
        if "num_tokens" not in row:
            raise ValueError(
                "DTR entry is missing num_tokens; regenerate DTR results with the current pipeline"
            )
        points.append(
            SequenceLengthPoint(
                doc_id=int(row["doc_id"]),
                repeat_index=int(row["repeat_index"]),
                dtr=float(row["dtr"]),
                response_length=int(row["num_tokens"]),
            )
        )

    points.sort(key=lambda point: (point.dtr, point.doc_id, point.repeat_index))
    return points


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


def build_title(
    run_dir: Path,
    task_name: str,
    model_name: str,
    user_title: str | None,
) -> str:
    if user_title is not None:
        return user_title
    return f"{run_dir.name} | {task_name} | {model_name} | DTR vs Response Length"


def write_summary_json(
    *,
    run_dir: Path,
    task_name: str,
    model_name: str,
    dtr_path: Path,
    results_path: Path,
    output_path: Path,
    points: list[SequenceLengthPoint],
    pearson: float,
) -> None:
    dtrs = [point.dtr for point in points]
    lengths = [point.response_length for point in points]
    summary = {
        "run_dir": str(run_dir),
        "task": task_name,
        "model": model_name,
        "dtr_path": str(dtr_path),
        "results_path": str(results_path),
        "num_sequences": len(points),
        "pearson_r": pearson,
        "dtr_min": min(dtrs),
        "dtr_max": max(dtrs),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": fmean(lengths),
        "points": [asdict(point) for point in points],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_summary(points: list[SequenceLengthPoint], pearson: float) -> None:
    dtrs = [point.dtr for point in points]
    lengths = [point.response_length for point in points]
    print(f"num_sequences: {len(points)}")
    print(f"dtr_range: [{min(dtrs):.4f}, {max(dtrs):.4f}]")
    print(f"length_range: [{min(lengths)}, {max(lengths)}]")
    print(f"length_mean: {fmean(lengths):.4f}")
    print(f"Pearson r: {pearson:.6f}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    dtr_path, results_path, output_plot, output_json = resolve_paths(args)

    points = load_points(dtr_path)
    if not points:
        raise ValueError(f"no DTR rows found in {dtr_path}")

    aggregated = (
        load_aggregated_results(run_dir)
        if args.results_path is None
        else json.loads(results_path.read_text(encoding="utf-8"))
    )
    task_name = infer_task_name(aggregated)
    model_name = resolve_model_name(aggregated, run_dir)
    pearson = pearson_r(
        [point.dtr for point in points],
        [float(point.response_length) for point in points],
    )

    if output_plot is not None:
        plot_to_png(
            points=points,
            pearson=pearson,
            output_path=output_plot,
            title=build_title(run_dir, task_name, model_name, args.title),
        )
    write_summary_json(
        run_dir=run_dir,
        task_name=task_name,
        model_name=model_name,
        dtr_path=dtr_path,
        results_path=results_path,
        output_path=output_json,
        points=points,
        pearson=pearson,
    )
    print_summary(points, pearson)
    print(f"Saved summary: {output_json}")
    if output_plot is not None:
        print(f"Saved plot: {output_plot}")


if __name__ == "__main__":
    main()
