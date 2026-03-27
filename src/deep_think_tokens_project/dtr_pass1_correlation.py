"""새 DTR JSON 파일명을 읽는 correlation 드라이버."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from src.deep_think_tokens_project.io import dtr_results_path
from src.deep_think_tokens_project.io import jsd_output_dir
from src.deep_think_tokens_project.io import latest_matching_file
from src.deep_think_tokens_project.io import load_aggregated_results
from src.deep_think_tokens_project.utils import DEFAULT_G
from src.deep_think_tokens_project.utils import DEFAULT_P
from src.deep_think_tokens_project.utils import compute_dtr_from_divergence_matrix
from src.plot.dtr_pass1_correlation import plot_to_png
from tasks.aime24.metrics import TASK_NAME, infer_task_name
from tasks.aime24.utils import resolve_model_identity, resolve_reasoning_tags, score_match

import torch


DEFAULT_OUTPUT_DIR_NAME = "dtr_pass1_correlation"


@dataclass(frozen=True)
class DtrRecord:
    """DTR JSON 한 행에서 correlation에 필요한 값만 보존한다."""

    dtr: float
    num_tokens: int | None = None


def prefix_suffix(prefix_len: int | None) -> str:
    if prefix_len is None:
        return ""
    return f"_prefix{prefix_len}"


def token_filter_suffix(
    start_token: int | None,
    end_token: int | None,
) -> str:
    if start_token is None and end_token is None:
        return ""
    if start_token is None:
        start_token = 0
    if end_token is not None:
        return f"_tokens{start_token}to{end_token - 1}"
    return f"_tokens{start_token}plus"


def artifact_suffix(
    prefix_len: int | None,
    start_token: int | None = None,
    end_token: int | None = None,
) -> str:
    return prefix_suffix(prefix_len) + token_filter_suffix(start_token, end_token)


def dtr_scope(
    prefix_len: int | None,
    start_token: int | None,
    end_token: int | None,
) -> str:
    if prefix_len is not None:
        return "prefix"
    if start_token is None and end_token is None:
        return "full"
    return "window"


def plot_filename(
    num_bins: int,
    prefix_len: int | None = None,
    start_token: int | None = None,
    end_token: int | None = None,
) -> str:
    return (
        "dtr_pass1_correlation"
        f"{artifact_suffix(prefix_len, start_token, end_token)}"
        f"_bins{num_bins}.png"
    )


def summary_filename(
    num_bins: int,
    prefix_len: int | None = None,
    start_token: int | None = None,
    end_token: int | None = None,
) -> str:
    return (
        "dtr_pass1_correlation"
        f"{artifact_suffix(prefix_len, start_token, end_token)}"
        f"_bins{num_bins}.json"
    )


@dataclass(frozen=True)
class SequenceResult:
    doc_id: int
    repeat_index: int
    dtr: float
    pass_at_1: float
    num_tokens: int | None = None


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
    parser.add_argument("--start-token", type=int)
    parser.add_argument("--end-token", type=int)
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
    start_token: int | None,
    end_token: int | None,
) -> Path | None:
    if output_plot is None:
        return None
    candidate = output_plot.resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / plot_filename(
            num_bins,
            prefix_len,
            start_token,
            end_token,
        )
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")


def validate_token_window(
    prefix_len: int | None,
    start_token: int | None,
    end_token: int | None,
    dtr_path: Path | None,
) -> None:
    if start_token is not None and start_token < 0:
        raise ValueError(f"start_token must be non-negative, got {start_token}")
    if end_token is not None and end_token < 1:
        raise ValueError(f"end_token must be >= 1, got {end_token}")
    if prefix_len is not None and (start_token is not None or end_token is not None):
        raise ValueError("cannot combine --prefix-len with --start-token/--end-token")
    if dtr_path is not None and (
        prefix_len is not None or start_token is not None or end_token is not None
    ):
        raise ValueError(
            "cannot use --dtr-path together with --prefix-len/--start-token/--end-token"
        )
    if (
        start_token is not None
        and end_token is not None
        and start_token >= end_token
    ):
        raise ValueError(
            f"start_token must be < end_token, got {start_token} >= {end_token}"
        )


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path | None, Path, Path, Path | None, Path]:
    run_dir = args.run_dir.resolve()
    start_token = getattr(args, "start_token", None)
    end_token = getattr(args, "end_token", None)
    validate_token_window(args.prefix_len, start_token, end_token, args.dtr_path)
    dtr_path = None
    if args.prefix_len is None and start_token is None and end_token is None:
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
        resolve_output_plot_path(
            args.output_plot,
            args.num_bins,
            args.prefix_len,
            start_token,
            end_token,
        )
        if args.output_plot is not None
        else output_dir
        / plot_filename(args.num_bins, args.prefix_len, start_token, end_token)
    )
    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else output_dir
        / summary_filename(args.num_bins, args.prefix_len, start_token, end_token)
    )
    return dtr_path, results_path, samples_path, output_plot, output_json


def load_dtr_by_key(path: Path) -> dict[tuple[int, int], float]:
    return {
        key: row.dtr for key, row in load_dtr_records_by_key(path).items()
    }


def load_dtr_records_by_key(path: Path) -> dict[tuple[int, int], DtrRecord]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        (int(row["doc_id"]), int(row["repeat_index"])): DtrRecord(
            dtr=float(row["dtr"]),
            num_tokens=(
                None if "num_tokens" not in row else int(row["num_tokens"])
            ),
        )
        for row in rows
    }


def load_prefix_dtr_by_key(
    run_dir: Path,
    *,
    prefix_len: int,
    g: float,
    p: float,
) -> dict[tuple[int, int], float]:
    return {
        key: row.dtr
        for key, row in load_prefix_dtr_records_by_key(
            run_dir,
            prefix_len=prefix_len,
            g=g,
            p=p,
        ).items()
    }


def load_prefix_dtr_records_by_key(
    run_dir: Path,
    *,
    prefix_len: int,
    g: float,
    p: float,
) -> dict[tuple[int, int], DtrRecord]:
    from src.deep_think_tokens_project.think_n import load_prefix_dtr_rows

    prefix_rows = load_prefix_dtr_rows(
        run_dir=run_dir,
        prefix_len=prefix_len,
        g=g,
        p=p,
    )
    return {
        key: DtrRecord(dtr=prefix_dtr, num_tokens=full_num_tokens)
        for key, (prefix_dtr, full_num_tokens) in prefix_rows.items()
    }


def load_window_dtr_records_by_key(
    run_dir: Path,
    *,
    start_token: int,
    end_token: int | None,
    g: float,
    p: float,
) -> dict[tuple[int, int], DtrRecord]:
    matrix_dir = jsd_output_dir(run_dir)
    matrix_paths = sorted(matrix_dir.glob("doc*_rep*.pt"))
    if not matrix_paths:
        raise FileNotFoundError(f"no divergence caches found under {matrix_dir}")

    window_dtr_by_key: dict[tuple[int, int], DtrRecord] = {}
    for matrix_path in matrix_paths:
        matrix_payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
        divergence_matrix = torch.as_tensor(matrix_payload["divergence_matrix"])
        stop = divergence_matrix.shape[1] if end_token is None else min(
            end_token, divergence_matrix.shape[1]
        )
        if start_token >= stop:
            raise ValueError(
                f"token window [{start_token}, {stop}) is empty for {matrix_path}"
            )
        window_matrix = divergence_matrix[:, start_token:stop]
        dtr_result = compute_dtr_from_divergence_matrix(window_matrix, g=g, p=p)
        key = (int(matrix_payload["doc_id"]), int(matrix_payload["repeat_index"]))
        window_dtr_by_key[key] = DtrRecord(
            dtr=float(dtr_result.dtr),
            num_tokens=int(matrix_payload["num_tokens"]),
        )
    return window_dtr_by_key


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
    num_tokens_by_key: dict[tuple[int, int], int | None] | None = None,
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
                        num_tokens=(
                            None
                            if num_tokens_by_key is None
                            else num_tokens_by_key[key]
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
    start_token: int | None = None,
    end_token: int | None = None,
) -> str:
    if user_title is not None:
        return user_title
    dtr_label = "DTR" if prefix_len is None else f"Prefix-{prefix_len} DTR"
    title = f"{run_dir.name} | {task_name} | {model_name} | {dtr_label} vs Pass@1"
    if start_token is None and end_token is None:
        return title
    window_start = 0 if start_token is None else start_token
    if end_token is None:
        return f"{title} | tokens {window_start}+"
    return f"{title} | tokens {window_start}-{end_token - 1}"


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
    start_token: int | None = None,
    end_token: int | None = None,
) -> None:
    token_lengths: list[int] = [
        row.num_tokens for row in rows if row.num_tokens is not None
    ]
    summary = {
        "run_dir": str(run_dir),
        "task": task_name,
        "model": model_name,
        "dtr_path": None if dtr_path is None else str(dtr_path),
        "dtr_scope": dtr_scope(prefix_len, start_token, end_token),
        "prefix_len": prefix_len,
        "start_token": start_token,
        "end_token": end_token,
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
    if token_lengths:
        summary["token_length_min"] = min(token_lengths)
        summary["token_length_max"] = max(token_lengths)
        summary["token_length_mean"] = fmean(token_lengths)
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
    if args.prefix_len is None and args.start_token is None and args.end_token is None:
        assert dtr_path is not None
        dtr_records_by_key = load_dtr_records_by_key(dtr_path)
    elif args.prefix_len is not None:
        dtr_records_by_key = load_prefix_dtr_records_by_key(
            run_dir,
            prefix_len=args.prefix_len,
            g=args.g,
            p=args.p,
        )
    else:
        dtr_records_by_key = load_window_dtr_records_by_key(
            run_dir,
            start_token=0 if args.start_token is None else args.start_token,
            end_token=args.end_token,
            g=args.g,
            p=args.p,
        )
    dtr_by_key = {key: row.dtr for key, row in dtr_records_by_key.items()}
    num_tokens_by_key = {
        key: row.num_tokens for key, row in dtr_records_by_key.items()
    }
    rows = load_sequence_results(
        dtr_by_key,
        samples_path,
        reasoning_tags=reasoning_tags,
        num_tokens_by_key=num_tokens_by_key,
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
                start_token=args.start_token,
                end_token=args.end_token,
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
        start_token=args.start_token,
        end_token=args.end_token,
    )
    print_summary(bins, binned_pearson)
    print(f"Saved summary: {output_json}")
    if output_plot is not None:
        print(f"Saved plot: {output_plot}")


if __name__ == "__main__":
    main()
