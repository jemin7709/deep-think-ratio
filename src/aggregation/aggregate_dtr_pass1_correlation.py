"""여러 run의 `dtr_pass1_correlation_bins*.json`을 bin index 기준으로 평균낸다."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from src.aggregation.dtr_pass1_correlation import BinSummary, pearson_r
from src.plot.dtr_pass1_correlation import plot_to_png


DEFAULT_INPUT_ROOT = Path("results")
DEFAULT_AGGREGATE_DIR_NAME = "dtr_pass1_correlation_aggregated"
SOURCE_GLOB = "dtr_pass1_correlation_bins*.json"


def aggregated_json_name(num_bins: int) -> str:
    return f"aggregated_dtr_pass1_correlation_bins{num_bins}.json"


def plot_summary_json_name(num_bins: int) -> str:
    return f"plot_dtr_pass1_correlation_bins{num_bins}_summary.json"


def plot_filename(num_bins: int) -> str:
    return f"dtr_pass1_correlation_bins{num_bins}.png"


@dataclass(frozen=True)
class SourceSummary:
    """개별 run에서 만든 binned correlation summary."""

    summary_path: Path
    run_dir: Path
    task: str
    model: str
    num_bins: int
    bins: list[BinSummary]


@dataclass(frozen=True)
class AggregatedBin:
    """여러 run에서 평균낸 최종 plot point."""

    bin_index: int
    source_count: int
    mean_source_bin_size: float
    mean_dtr: float
    pass_at_1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate run-level dtr_pass1_correlation_bins*.json summaries."
    )
    parser.add_argument("input_root", nargs="?", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--aggregate-dir-name", default=DEFAULT_AGGREGATE_DIR_NAME)
    parser.add_argument("--title")
    return parser.parse_args()


def discover_summary_paths(input_root: Path, aggregate_dir_name: str) -> list[Path]:
    excluded_dir = input_root / aggregate_dir_name
    paths = [
        path
        for path in sorted(input_root.rglob(SOURCE_GLOB))
        if excluded_dir not in path.parents
    ]
    if not paths:
        raise FileNotFoundError(f"no {SOURCE_GLOB} found under {input_root}")
    return paths


def load_source_summary(summary_path: Path) -> SourceSummary:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    bins = [
        BinSummary(
            bin_index=int(entry["bin_index"]),
            count=int(entry["count"]),
            rank_start=int(entry["rank_start"]),
            rank_end=int(entry["rank_end"]),
            dtr_min=float(entry["dtr_min"]),
            dtr_max=float(entry["dtr_max"]),
            mean_dtr=float(entry["mean_dtr"]),
            pass_at_1=float(entry["pass_at_1"]),
        )
        for entry in payload["bins"]
    ]
    return SourceSummary(
        summary_path=summary_path,
        run_dir=Path(payload["run_dir"]),
        task=str(payload["task"]),
        model=str(payload["model"]),
        num_bins=int(payload["num_bins"]),
        bins=bins,
    )


def load_source_summaries(summary_paths: list[Path]) -> list[SourceSummary]:
    summaries = [load_source_summary(path) for path in summary_paths]
    if not summaries:
        raise ValueError("need at least one source summary")

    first = summaries[0]
    expected_indices = [index + 1 for index in range(first.num_bins)]

    for summary in summaries[1:]:
        if summary.task != first.task or summary.model != first.model:
            raise ValueError("all source summaries must share task and model")
        if summary.num_bins != first.num_bins:
            raise ValueError("all source summaries must share num_bins")

    for summary in summaries:
        actual_indices = [entry.bin_index for entry in summary.bins]
        if actual_indices != expected_indices:
            raise ValueError(
                f"unexpected bin indices in {summary.summary_path}: {actual_indices}"
            )

    return summaries


def aggregate_bins(summaries: list[SourceSummary]) -> list[AggregatedBin]:
    aggregated: list[AggregatedBin] = []
    for per_bin_entries in zip(*(summary.bins for summary in summaries), strict=True):
        aggregated.append(
            AggregatedBin(
                bin_index=per_bin_entries[0].bin_index,
                source_count=len(per_bin_entries),
                mean_source_bin_size=fmean(entry.count for entry in per_bin_entries),
                mean_dtr=fmean(entry.mean_dtr for entry in per_bin_entries),
                pass_at_1=fmean(entry.pass_at_1 for entry in per_bin_entries),
            )
        )
    return aggregated


def build_plot_bins(aggregated_bins: list[AggregatedBin]) -> list[BinSummary]:
    rank_start = 1
    plot_bins: list[BinSummary] = []
    for entry in aggregated_bins:
        rounded_count = max(1, round(entry.mean_source_bin_size))
        rank_end = rank_start + rounded_count - 1
        plot_bins.append(
            BinSummary(
                bin_index=entry.bin_index,
                count=rounded_count,
                rank_start=rank_start,
                rank_end=rank_end,
                dtr_min=entry.mean_dtr,
                dtr_max=entry.mean_dtr,
                mean_dtr=entry.mean_dtr,
                pass_at_1=entry.pass_at_1,
            )
        )
        rank_start = rank_end + 1
    return plot_bins


def write_plot_summary_json(
    *,
    input_root: Path,
    aggregate_dir: Path,
    summaries: list[SourceSummary],
    aggregated_bins: list[AggregatedBin],
    output_path: Path,
) -> None:
    payload = {
        "input_root": str(input_root),
        "aggregate_dir": str(aggregate_dir),
        "task": summaries[0].task,
        "model": summaries[0].model,
        "source_count": len(summaries),
        "num_bins": len(aggregated_bins),
        "pearson_r_binned": pearson_r(
            [entry.mean_dtr for entry in aggregated_bins],
            [entry.pass_at_1 for entry in aggregated_bins],
        ),
        "bins": [asdict(entry) for entry in aggregated_bins],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_aggregated_json(
    *,
    input_root: Path,
    aggregate_dir: Path,
    summaries: list[SourceSummary],
    aggregated_bins: list[AggregatedBin],
    plot_path: Path,
    plot_summary_path: Path,
) -> Path:
    output_path = aggregate_dir / aggregated_json_name(len(aggregated_bins))
    payload = {
        "input_root": str(input_root),
        "aggregate_dir": str(aggregate_dir),
        "task": summaries[0].task,
        "model": summaries[0].model,
        "source_count": len(summaries),
        "source_paths": [str(summary.summary_path) for summary in summaries],
        "run_dirs": [str(summary.run_dir) for summary in summaries],
        "num_bins": len(aggregated_bins),
        "pearson_r_binned": pearson_r(
            [entry.mean_dtr for entry in aggregated_bins],
            [entry.pass_at_1 for entry in aggregated_bins],
        ),
        "bins": [asdict(entry) for entry in aggregated_bins],
        "plot_path": str(plot_path),
        "plot_summary_path": str(plot_summary_path),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_title(
    summaries: list[SourceSummary], source_count: int, user_title: str | None
) -> str:
    if user_title is not None:
        return user_title
    first = summaries[0]
    return (
        f"{first.task} | {first.model} | Averaged DTR vs Pass@1 ({source_count} runs)"
    )


def print_summary(aggregated_bins: list[AggregatedBin], aggregated_json: Path) -> None:
    print("bin  runs  mean_dtr  pass@1")
    for entry in aggregated_bins:
        print(
            f"{entry.bin_index:>3}  "
            f"{entry.source_count:>4}  "
            f"{entry.mean_dtr:>8.4f}  "
            f"{entry.pass_at_1:>6.4f}"
        )
    print(f"Saved aggregated summary: {aggregated_json}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    aggregate_dir = input_root / args.aggregate_dir_name

    summary_paths = discover_summary_paths(input_root, args.aggregate_dir_name)
    summaries = load_source_summaries(summary_paths)
    aggregated_bins = aggregate_bins(summaries)
    plot_path = aggregate_dir / plot_filename(len(aggregated_bins))
    plot_summary_path = aggregate_dir / plot_summary_json_name(len(aggregated_bins))
    plot_bins = build_plot_bins(aggregated_bins)
    binned_pearson = pearson_r(
        [entry.mean_dtr for entry in aggregated_bins],
        [entry.pass_at_1 for entry in aggregated_bins],
    )

    aggregate_dir.mkdir(parents=True, exist_ok=True)
    plot_to_png(
        bins=plot_bins,
        pearson=binned_pearson,
        output_path=plot_path,
        title=build_title(summaries, len(summaries), args.title),
    )
    write_plot_summary_json(
        input_root=input_root,
        aggregate_dir=aggregate_dir,
        summaries=summaries,
        aggregated_bins=aggregated_bins,
        output_path=plot_summary_path,
    )
    aggregated_json = write_aggregated_json(
        input_root=input_root,
        aggregate_dir=aggregate_dir,
        summaries=summaries,
        aggregated_bins=aggregated_bins,
        plot_path=plot_path,
        plot_summary_path=plot_summary_path,
    )
    print_summary(aggregated_bins, aggregated_json)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
