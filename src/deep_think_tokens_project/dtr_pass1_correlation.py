"""새 DTR JSON 파일명을 읽는 correlation 드라이버."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from src.deep_think_tokens_project.io import dtr_results_path
from src.deep_think_tokens_project.io import latest_matching_file
from src.deep_think_tokens_project.io import load_aggregated_results
from src.deep_think_tokens_project.utils import DEFAULT_G
from src.deep_think_tokens_project.utils import DEFAULT_P
from src.plot.dtr_pass1_correlation import plot_to_png
from tasks.aime24.metrics import TASK_NAME, infer_task_name
from tasks.aime24.utils import resolve_model_identity, resolve_reasoning_tags, score_match


DEFAULT_OUTPUT_DIR_NAME = "dtr_pass1_correlation"


def artifact_suffix(prefix_len: int | None) -> str:
    if prefix_len is None:
        return ""
    return f"_prefix{prefix_len}"


def dtr_scope(prefix_len: int | None) -> str:
    return "full" if prefix_len is None else "prefix"


def plot_filename(num_bins: int, prefix_len: int | None = None) -> str:
    return f"dtr_pass1_correlation{artifact_suffix(prefix_len)}_bins{num_bins}.png"


def summary_filename(num_bins: int, prefix_len: int | None = None) -> str:
    return f"dtr_pass1_correlation{artifact_suffix(prefix_len)}_bins{num_bins}.json"


@dataclass(frozen=True)
class SequenceResult:
    doc_id: int
    repeat_index: int
    dtr: float
    pass_at_1: float


@dataclass(frozen=True)
class BinSummary:
    bin_index: int
    count: int
    rank_start: int
    rank_end: int
    dtr_min: float
    dtr_max: float
    mean_dtr: float
    pass_at_1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the deep-think-tokens DTR vs Pass@1 correlation."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dtr-path", type=Path)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--samples-path", type=Path)
    parser.add_argument("--prefix-len", type=int)
    parser.add_argument("--g", type=float, default=DEFAULT_G)
    parser.add_argument("--p", type=float, default=DEFAULT_P)
    parser.add_argument("--num-bins", type=int, default=5)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--title")
    return parser.parse_args()


def default_output_dir(run_dir: Path) -> Path:
    return run_dir / DEFAULT_OUTPUT_DIR_NAME


def resolve_output_plot_path(
    output_plot: Path | None,
    num_bins: int,
    prefix_len: int | None,
) -> Path | None:
    if output_plot is None:
        return None
    candidate = output_plot.resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / plot_filename(num_bins, prefix_len)
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path | None, Path, Path, Path | None, Path]:
    run_dir = args.run_dir.resolve()
    if args.dtr_path is not None and args.prefix_len is not None:
        raise ValueError("cannot use --dtr-path together with --prefix-len")
    dtr_path = None
    if args.prefix_len is None:
        dtr_path = (
            args.dtr_path.resolve()
            if args.dtr_path is not None
            else dtr_results_path(run_dir, g=args.g, p=args.p)
        )
    results_path = (
        args.results_path.resolve()
        if args.results_path is not None
        else latest_matching_file(run_dir, "results_*.json")
    )
    if args.samples_path is not None:
        samples_path = args.samples_path.resolve()
    else:
        aggregated = json.loads(results_path.read_text(encoding="utf-8"))
        task_name = infer_task_name(aggregated)
        samples_path = latest_matching_file(run_dir, f"samples_{task_name}_*.jsonl")

    output_dir = default_output_dir(run_dir)
    output_plot = (
        resolve_output_plot_path(args.output_plot, args.num_bins, args.prefix_len)
        if args.output_plot is not None
        else output_dir / plot_filename(args.num_bins, args.prefix_len)
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else output_dir / summary_filename(args.num_bins, args.prefix_len)
    )
    return dtr_path, results_path, samples_path, output_plot, output_json


def load_dtr_by_key(path: Path) -> dict[tuple[int, int], float]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(row["doc_id"]), int(row["repeat_index"])): float(row["dtr"])
        for row in rows
    }


def load_prefix_dtr_by_key(
    run_dir: Path,
    *,
    prefix_len: int,
    g: float,
    p: float,
) -> dict[tuple[int, int], float]:
    from src.deep_think_tokens_project.think_n import load_prefix_dtr_rows

    prefix_rows = load_prefix_dtr_rows(
        run_dir=run_dir,
        prefix_len=prefix_len,
        g=g,
        p=p,
    )
    return {key: prefix_dtr for key, (prefix_dtr, _num_tokens) in prefix_rows.items()}


def validate_supported_task(task_name: str) -> None:
    if task_name != TASK_NAME:
        raise NotImplementedError(
            f"DTR/pass@1 plotting currently supports only {TASK_NAME}, got {task_name}"
        )


def load_sequence_results(
    dtr_by_key: dict[tuple[int, int], float],
    samples_path: Path,
    *,
    reasoning_tags: list[tuple[str, str]] | None,
) -> list[SequenceResult]:
    rows: list[SequenceResult] = []
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
                rows.append(
                    SequenceResult(
                        doc_id=doc_id,
                        repeat_index=repeat_index,
                        dtr=dtr_by_key[key],
                        pass_at_1=score_match(
                            target,
                            response,
                            reasoning_tags=reasoning_tags,
                        ),
                    )
                )
                seen_keys.add(key)

    missing_from_samples = sorted(set(dtr_by_key) - seen_keys)
    if missing_from_samples:
        raise ValueError(
            "DTR entries without matching sample rows: "
            + ", ".join(f"{doc}:{repeat}" for doc, repeat in missing_from_samples[:5])
        )
    rows.sort(key=lambda item: item.dtr)
    return rows


def make_bins(rows: list[SequenceResult], num_bins: int) -> list[BinSummary]:
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if len(rows) < num_bins:
        raise ValueError(
            f"need at least {num_bins} rows for {num_bins} bins, got {len(rows)}"
        )

    base_size, remainder = divmod(len(rows), num_bins)
    bins: list[BinSummary] = []
    start = 0
    for bin_offset in range(num_bins):
        size = base_size + (1 if bin_offset < remainder else 0)
        chunk = rows[start : start + size]
        bins.append(
            BinSummary(
                bin_index=bin_offset + 1,
                count=len(chunk),
                rank_start=start + 1,
                rank_end=start + len(chunk),
                dtr_min=chunk[0].dtr,
                dtr_max=chunk[-1].dtr,
                mean_dtr=fmean(row.dtr for row in chunk),
                pass_at_1=fmean(row.pass_at_1 for row in chunk),
            )
        )
        start += size
    return bins


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
    prefix_len: int | None = None,
) -> str:
    if user_title is not None:
        return user_title
    dtr_label = "DTR" if prefix_len is None else f"Prefix-{prefix_len} DTR"
    return f"{run_dir.name} | {task_name} | {model_name} | {dtr_label} vs Pass@1"


def write_summary_json(
    *,
    run_dir: Path,
    task_name: str,
    model_name: str,
    dtr_path: Path | None,
    results_path: Path,
    samples_path: Path,
    output_path: Path,
    rows: list[SequenceResult],
    bins: list[BinSummary],
    binned_pearson: float,
    prefix_len: int | None = None,
) -> None:
    summary = {
        "run_dir": str(run_dir),
        "task": task_name,
        "model": model_name,
        "dtr_path": None if dtr_path is None else str(dtr_path),
        "dtr_scope": dtr_scope(prefix_len),
        "prefix_len": prefix_len,
        "results_path": str(results_path),
        "samples_path": str(samples_path),
        "num_sequences": len(rows),
        "num_bins": len(bins),
        "pearson_r_binned": binned_pearson,
        "pearson_r_sequence_level": pearson_r(
            [row.dtr for row in rows],
            [row.pass_at_1 for row in rows],
        ),
        "bins": [asdict(entry) for entry in bins],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def print_summary(bins: list[BinSummary], pearson: float) -> None:
    print("bin  count  mean_dtr  pass@1  dtr_range")
    for entry in bins:
        print(
            f"{entry.bin_index:>3}  "
            f"{entry.count:>5}  "
            f"{entry.mean_dtr:>8.4f}  "
            f"{entry.pass_at_1:>6.4f}  "
            f"[{entry.dtr_min:.4f}, {entry.dtr_max:.4f}]"
        )
    print(f"Pearson r (binned): {pearson:.6f}")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    dtr_path, results_path, samples_path, output_plot, output_json = resolve_paths(args)
    aggregated = (
        load_aggregated_results(run_dir)
        if args.results_path is None
        else json.loads(results_path.read_text(encoding="utf-8"))
    )
    task_name = infer_task_name(aggregated)
    validate_supported_task(task_name)
    model_name = resolve_model_identity(aggregated, run_dir)
    reasoning_tags = resolve_reasoning_tags(aggregated)
    if args.prefix_len is None:
        assert dtr_path is not None
        dtr_by_key = load_dtr_by_key(dtr_path)
    else:
        dtr_by_key = load_prefix_dtr_by_key(
            run_dir,
            prefix_len=args.prefix_len,
            g=args.g,
            p=args.p,
        )
    rows = load_sequence_results(
        dtr_by_key,
        samples_path,
        reasoning_tags=reasoning_tags,
    )
    bins = make_bins(rows, args.num_bins)
    binned_pearson = pearson_r(
        [entry.mean_dtr for entry in bins],
        [entry.pass_at_1 for entry in bins],
    )

    if output_plot is not None:
        plot_to_png(
            bins=bins,
            pearson=binned_pearson,
            output_path=output_plot,
            title=build_title(
                run_dir,
                task_name,
                model_name,
                args.title,
                prefix_len=args.prefix_len,
            ),
        )
    write_summary_json(
        run_dir=run_dir,
        task_name=task_name,
        model_name=model_name,
        dtr_path=dtr_path,
        results_path=results_path,
        samples_path=samples_path,
        output_path=output_json,
        rows=rows,
        bins=bins,
        binned_pearson=binned_pearson,
        prefix_len=args.prefix_len,
    )
    print_summary(bins, binned_pearson)
    print(f"Saved summary: {output_json}")
    if output_plot is not None:
        print(f"Saved plot: {output_plot}")


if __name__ == "__main__":
    main()
