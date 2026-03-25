"""`deep-think-tokens` prefix DTR로 Think@n 실험을 재현한다."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

import torch

from src.deep_think_tokens_project.io import load_aggregated_results
from src.deep_think_tokens_project.io import load_sample_rows
from src.deep_think_tokens_project.io import jsd_output_dir
from src.deep_think_tokens_project.utils import DEFAULT_G
from src.deep_think_tokens_project.utils import DEFAULT_P
from src.deep_think_tokens_project.utils import compute_dtr_from_divergence_matrix
from src.deep_think_tokens_project.utils import validate_prefix_len
from src.experiment.repetition_metrics import mean_seq_rep_n


REP_N_VALUES = (2, 4)
REP_LEVELS = ("token", "word")


@dataclass(frozen=True)
class RepeatRecord:
    repeat_index: int
    prefix_dtr: float
    full_num_tokens: int
    rank: int
    selected: bool


@dataclass(frozen=True)
class DocResult:
    doc_id: int
    target: str
    selected_repeat_indices: list[int]
    ranked_repeats: list[RepeatRecord]
    metrics: dict[str, float]
    cost: dict[str, int]


def build_repetition_metrics(
    *,
    selected_completions: list[str],
    completions: list[str],
    model_name: str,
    reasoning_tags: list[tuple[str, str]] | None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for n in REP_N_VALUES:
        for level in REP_LEVELS:
            metrics[f"selected_{level}_rep_{n}"] = mean_seq_rep_n(
                selected_completions,
                n=n,
                granularity=level,
                model_name=model_name,
                reasoning_tags=reasoning_tags,
                strip_reasoning=False,
            )
            metrics[f"full_{level}_rep_{n}"] = mean_seq_rep_n(
                completions,
                n=n,
                granularity=level,
                model_name=model_name,
                reasoning_tags=reasoning_tags,
                strip_reasoning=False,
            )
    return metrics


def resolve_selected_count(
    *,
    repeats: int,
    top_fraction: float,
    selected_count: int | None,
) -> int:
    if selected_count is not None:
        if selected_count < 1 or selected_count > repeats:
            raise ValueError(
                f"selected_count must be between 1 and repeats ({repeats}), got {selected_count}"
            )
        return selected_count
    if top_fraction <= 0.0 or top_fraction > 1.0:
        raise ValueError(
            f"top_fraction must be in the interval (0, 1], got {top_fraction}"
        )
    return max(1, math.ceil(repeats * top_fraction))


def experiment_slug(
    *,
    prefix_len: int,
    repeats: int,
    selected_count: int,
    g: float,
    p: float,
) -> str:
    return (
        f"prefix{prefix_len}_top{selected_count}of{repeats}"
        f"_g{format(g, 'g')}_p{format(p, 'g')}"
    )


def build_output_dir(
    *,
    run_dir: Path,
    prefix_len: int,
    repeats: int,
    selected_count: int,
    g: float,
    p: float,
) -> Path:
    return (
        run_dir.resolve()
        / "experiments"
        / experiment_slug(
            prefix_len=prefix_len,
            repeats=repeats,
            selected_count=selected_count,
            g=g,
            p=p,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a run with prefix deep-think-tokens DTR ranking."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prefix-len", type=int, default=50)
    parser.add_argument("--top-fraction", type=float, default=0.5)
    parser.add_argument("--selected-count", type=int)
    parser.add_argument("--g", type=float, default=DEFAULT_G)
    parser.add_argument("--p", type=float, default=DEFAULT_P)
    return parser.parse_args()


def load_prefix_dtr_rows(
    *,
    run_dir: Path,
    prefix_len: int,
    g: float,
    p: float,
) -> dict[tuple[int, int], tuple[float, int]]:
    validate_prefix_len(prefix_len)
    matrix_dir = jsd_output_dir(run_dir)
    matrix_paths = sorted(matrix_dir.glob("doc*_rep*.pt"))
    if not matrix_paths:
        raise FileNotFoundError(f"no divergence caches found under {matrix_dir}")

    prefix_dtr_by_key: dict[tuple[int, int], tuple[float, int]] = {}
    for matrix_path in matrix_paths:
        payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
        divergence_matrix = torch.as_tensor(payload["divergence_matrix"])
        prefix_matrix = divergence_matrix[:, : min(prefix_len, divergence_matrix.shape[1])]
        dtr_result = compute_dtr_from_divergence_matrix(prefix_matrix, g=g, p=p)
        key = (int(payload["doc_id"]), int(payload["repeat_index"]))
        prefix_dtr_by_key[key] = (float(dtr_result.dtr), int(payload["num_tokens"]))
    return prefix_dtr_by_key


def build_ranked_repeats(
    *,
    doc_id: int,
    completions: list[str],
    prefix_rows: dict[tuple[int, int], tuple[float, int]],
    selected_count: int,
) -> list[RepeatRecord]:
    ranking_basis = []
    for repeat_index, _completion in enumerate(completions):
        key = (doc_id, repeat_index)
        if key not in prefix_rows:
            raise ValueError(
                f"missing prefix DTR for doc_id={doc_id}, repeat_index={repeat_index}"
            )
        prefix_dtr, full_num_tokens = prefix_rows[key]
        ranking_basis.append((repeat_index, prefix_dtr, full_num_tokens))

    ranking_basis.sort(key=lambda item: (-item[1], item[0]))
    selected_repeat_indices = {
        repeat_index for repeat_index, _, _ in ranking_basis[:selected_count]
    }
    return [
        RepeatRecord(
            repeat_index=repeat_index,
            prefix_dtr=prefix_dtr,
            full_num_tokens=full_num_tokens,
            rank=rank,
            selected=repeat_index in selected_repeat_indices,
        )
        for rank, (repeat_index, prefix_dtr, full_num_tokens) in enumerate(
            ranking_basis, start=1
        )
    ]


def build_doc_result(
    *,
    row: dict,
    selected_count: int,
    prefix_len: int,
    prefix_rows: dict[tuple[int, int], tuple[float, int]],
    reasoning_tags: list[tuple[str, str]] | None,
    model_name: str,
) -> DocResult:
    from tasks.aime24.utils import score_avg_at_n, score_maj_at_n

    doc_id = int(row["doc_id"])
    target = str(row["target"])
    completions = [str(response) for response in row["resps"][0]]
    repeats = len(completions)
    ranked_repeats = build_ranked_repeats(
        doc_id=doc_id,
        completions=completions,
        prefix_rows=prefix_rows,
        selected_count=selected_count,
    )
    selected_repeat_indices = [
        record.repeat_index for record in ranked_repeats if record.selected
    ]
    selected_completions = [completions[index] for index in selected_repeat_indices]

    full_cost = sum(record.full_num_tokens for record in ranked_repeats)
    prefix_cost = sum(
        min(prefix_len, record.full_num_tokens) for record in ranked_repeats
    )
    continuation_cost = sum(
        max(record.full_num_tokens - prefix_len, 0)
        for record in ranked_repeats
        if record.selected
    )
    think_cost = prefix_cost + continuation_cost

    metrics = {
        f"think_maj@{selected_count}": score_maj_at_n(
            target,
            selected_completions,
            n=selected_count,
            reasoning_tags=reasoning_tags,
        ),
        f"cons_maj@{repeats}": score_maj_at_n(
            target,
            completions,
            n=repeats,
            reasoning_tags=reasoning_tags,
        ),
        f"mean_avg@{repeats}": score_avg_at_n(
            target,
            completions,
            n=repeats,
            reasoning_tags=reasoning_tags,
        ),
    }
    metrics.update(
        build_repetition_metrics(
            selected_completions=selected_completions,
            completions=completions,
            model_name=model_name,
            reasoning_tags=reasoning_tags,
        )
    )

    return DocResult(
        doc_id=doc_id,
        target=target,
        selected_repeat_indices=selected_repeat_indices,
        ranked_repeats=ranked_repeats,
        metrics=metrics,
        cost={
            "full_tokens": full_cost,
            "prefix_tokens": prefix_cost,
            "continuation_tokens": continuation_cost,
            "think_tokens": think_cost,
        },
    )


def summarize_doc_results(
    *,
    doc_results: list[DocResult],
    repeats: int,
    selected_count: int,
) -> dict:
    think_key = f"think_maj@{selected_count}"
    cons_key = f"cons_maj@{repeats}"
    mean_key = f"mean_avg@{repeats}"
    metric_keys = [think_key, cons_key, mean_key]
    metric_keys.extend(
        f"{scope}_{level}_rep_{n}"
        for scope in ("selected", "full")
        for level in REP_LEVELS
        for n in REP_N_VALUES
    )

    metrics = {
        key: fmean(result.metrics[key] for result in doc_results) for key in metric_keys
    }
    metrics["num_docs"] = len(doc_results)
    total_full_tokens = sum(result.cost["full_tokens"] for result in doc_results)
    total_think_tokens = sum(result.cost["think_tokens"] for result in doc_results)
    saved_tokens = total_full_tokens - total_think_tokens
    return {
        "metrics": metrics,
        "cost": {
            "total_full_tokens": total_full_tokens,
            "total_think_tokens": total_think_tokens,
            "mean_full_tokens_per_doc": total_full_tokens / len(doc_results),
            "mean_think_tokens_per_doc": total_think_tokens / len(doc_results),
            "saved_tokens": saved_tokens,
            "saved_pct": saved_tokens / total_full_tokens if total_full_tokens else 0.0,
        },
        "delta": {
            "vs_cons_maj": metrics[think_key] - metrics[cons_key],
            "vs_mean_avg": metrics[think_key] - metrics[mean_key],
        },
    }


def render_summary(
    *,
    run_dir: Path,
    output_dir: Path,
    task_name: str,
    model_name: str,
    prefix_len: int,
    selected_count: int,
    repeats: int,
    p: float,
    summary: dict,
) -> str:
    think_key = f"think_maj@{selected_count}"
    cons_key = f"cons_maj@{repeats}"
    mean_key = f"mean_avg@{repeats}"
    lines = [
        f"run_dir: {run_dir}",
        f"output_dir: {output_dir}",
        f"task: {task_name}",
        f"model: {model_name}",
        f"prefix_len: {prefix_len}",
        f"selected_count: {selected_count}",
        f"p: {p}",
        f"{think_key}: {summary['metrics'][think_key]:.6f}",
        f"{cons_key}: {summary['metrics'][cons_key]:.6f}",
        f"{mean_key}: {summary['metrics'][mean_key]:.6f}",
        f"delta_vs_cons_maj: {summary['delta']['vs_cons_maj']:.6f}",
        f"delta_vs_mean_avg: {summary['delta']['vs_mean_avg']:.6f}",
        f"total_full_tokens: {summary['cost']['total_full_tokens']}",
        f"total_think_tokens: {summary['cost']['total_think_tokens']}",
        f"mean_full_tokens_per_doc: {summary['cost']['mean_full_tokens_per_doc']:.6f}",
        f"mean_think_tokens_per_doc: {summary['cost']['mean_think_tokens_per_doc']:.6f}",
        f"saved_tokens: {summary['cost']['saved_tokens']}",
        f"saved_pct: {summary['cost']['saved_pct']:.6%}",
    ]
    for scope in ("selected", "full"):
        for level in REP_LEVELS:
            for n in REP_N_VALUES:
                metric_key = f"{scope}_{level}_rep_{n}"
                lines.append(f"{metric_key}: {summary['metrics'][metric_key]:.6f}")
    lines.append(f"num_docs: {summary['metrics']['num_docs']}")
    return "\n".join(lines)


def run_experiment(
    *,
    run_dir: Path,
    prefix_len: int = 50,
    top_fraction: float = 0.5,
    selected_count: int | None = None,
    g: float = DEFAULT_G,
    p: float = DEFAULT_P,
) -> tuple[Path, Path]:
    from tasks.aime24.metrics import infer_repeats, infer_task_name
    from tasks.aime24.utils import resolve_model_identity, resolve_reasoning_tags

    resolved_run_dir = run_dir.resolve()
    aggregated = load_aggregated_results(resolved_run_dir)
    task_name = infer_task_name(aggregated)
    repeats = infer_repeats(aggregated, task_name)
    chosen_count = resolve_selected_count(
        repeats=repeats,
        top_fraction=top_fraction,
        selected_count=selected_count,
    )
    reasoning_tags = resolve_reasoning_tags(aggregated)
    model_name = resolve_model_identity(aggregated, resolved_run_dir)
    sample_rows = load_sample_rows(resolved_run_dir, task_name)
    prefix_rows = load_prefix_dtr_rows(
        run_dir=resolved_run_dir,
        prefix_len=prefix_len,
        g=g,
        p=p,
    )
    doc_results = [
        build_doc_result(
            row=row,
            selected_count=chosen_count,
            prefix_len=prefix_len,
            prefix_rows=prefix_rows,
            reasoning_tags=reasoning_tags,
            model_name=model_name,
        )
        for row in sample_rows
    ]
    if not doc_results:
        raise ValueError(f"no sample rows found under {resolved_run_dir}")

    summary = summarize_doc_results(
        doc_results=doc_results,
        repeats=repeats,
        selected_count=chosen_count,
    )
    output_dir = build_output_dir(
        run_dir=resolved_run_dir,
        prefix_len=prefix_len,
        repeats=repeats,
        selected_count=chosen_count,
        g=g,
        p=p,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_dir": str(resolved_run_dir),
        "output_dir": str(output_dir),
        "task": task_name,
        "model": model_name,
        "repeats": repeats,
        "selected_count": chosen_count,
        "prefix_len": prefix_len,
        "g": g,
        "p": p,
        "rho": p,
        "summary": summary,
        "docs": [
            {
                "doc_id": result.doc_id,
                "target": result.target,
                "selected_repeat_indices": result.selected_repeat_indices,
                "ranked_repeats": [asdict(record) for record in result.ranked_repeats],
                "metrics": result.metrics,
                "cost": result.cost,
            }
            for result in doc_results
        ],
    }
    summary_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"
    summary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_txt.write_text(
        render_summary(
            run_dir=resolved_run_dir,
            output_dir=output_dir,
            task_name=task_name,
            model_name=model_name,
            prefix_len=prefix_len,
            selected_count=chosen_count,
            repeats=repeats,
            p=p,
            summary=summary,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_json, summary_txt


def main() -> None:
    args = parse_args()
    summary_json, summary_txt = run_experiment(
        run_dir=args.run_dir,
        prefix_len=args.prefix_len,
        top_fraction=args.top_fraction,
        selected_count=args.selected_count,
        g=args.g,
        p=args.p,
    )
    print(summary_json)
    print(summary_txt.read_text(encoding="utf-8").rstrip())


if __name__ == "__main__":
    main()
