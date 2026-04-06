"""`deep-think-tokens` prefix DTR 하위 선택 실험."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from src.deep_think_tokens_project.io import load_aggregated_results
from src.deep_think_tokens_project.think_n import REP_LEVELS
from src.deep_think_tokens_project.think_n import REP_N_VALUES
from src.deep_think_tokens_project.think_n import DEFAULT_G
from src.deep_think_tokens_project.think_n import DEFAULT_P
from src.deep_think_tokens_project.think_n import DocResult
from src.deep_think_tokens_project.think_n import RepeatRecord
from src.deep_think_tokens_project.think_n import build_cost_definition
from src.deep_think_tokens_project.think_n import build_selection_stats
from src.deep_think_tokens_project.think_n import build_repetition_metrics
from src.deep_think_tokens_project.think_n import load_prefix_dtr_rows
from src.deep_think_tokens_project.think_n import load_sample_rows


def resolve_selected_count(
    *,
    repeats: int,
    bottom_fraction: float,
    selected_count: int | None,
) -> int:
    if selected_count is not None:
        if selected_count < 1 or selected_count > repeats:
            raise ValueError(
                f"selected_count must be between 1 and repeats ({repeats}), got {selected_count}"
            )
        return selected_count
    if bottom_fraction <= 0.0 or bottom_fraction > 1.0:
        raise ValueError(
            f"bottom_fraction must be in the interval (0, 1], got {bottom_fraction}"
        )
    return max(1, math.ceil(repeats * bottom_fraction))


def experiment_slug(
    *,
    prefix_len: int,
    repeats: int,
    selected_count: int,
    g: float,
    p: float,
) -> str:
    return (
        f"prefix{prefix_len}_bottom{selected_count}of{repeats}"
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
        description="Replay a run with lower prefix deep-think-tokens DTR ranking."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--prefix-len", type=int, default=50)
    parser.add_argument("--bottom-fraction", type=float, default=0.5)
    parser.add_argument("--selected-count", type=int)
    parser.add_argument("--g", type=float, default=DEFAULT_G)
    parser.add_argument("--p", type=float, default=DEFAULT_P)
    return parser.parse_args()


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
        repeat_index for repeat_index, _, _ in ranking_basis[-selected_count:]
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
    from tasks.aime24.utils import score_avg_at_n, score_maj_at_n, score_pass_at_k

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

    full_prefix_cost = sum(
        min(prefix_len, record.full_num_tokens) for record in ranked_repeats
    )
    full_completion_cost = sum(record.full_num_tokens for record in ranked_repeats)
    full_cost = full_prefix_cost + full_completion_cost
    prefix_cost = sum(
        min(prefix_len, record.full_num_tokens)
        for record in ranked_repeats
        if record.selected
    )
    completion_cost = sum(
        record.full_num_tokens for record in ranked_repeats if record.selected
    )
    bottom_cost = prefix_cost + completion_cost

    bottom_maj = score_maj_at_n(
        target,
        selected_completions,
        n=selected_count,
        reasoning_tags=reasoning_tags,
    )
    bottom_pass = score_pass_at_k(
        target,
        selected_completions,
        n=selected_count,
        k=1,
        reasoning_tags=reasoning_tags,
    )
    cons_pass = score_pass_at_k(
        target,
        completions,
        n=repeats,
        k=1,
        reasoning_tags=reasoning_tags,
    )
    metrics = {
        "bottom_pass@1": bottom_pass,
        f"bottom_maj@{selected_count}": bottom_maj,
        "cons_pass@1": cons_pass,
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
        selection_stats=build_selection_stats(
            ranked_repeats=ranked_repeats,
            selected_majority_score=bottom_maj,
        ),
        metrics=metrics,
        cost={
            "full_tokens": full_cost,
            "prefix_tokens": prefix_cost,
            "completion_tokens": completion_cost,
            "bottom_tokens": bottom_cost,
        },
    )


def summarize_doc_results(
    *,
    doc_results: list[DocResult],
    repeats: int,
    selected_count: int,
) -> dict:
    bottom_pass_key = "bottom_pass@1"
    bottom_key = f"bottom_maj@{selected_count}"
    cons_pass_key = "cons_pass@1"
    cons_key = f"cons_maj@{repeats}"
    mean_key = f"mean_avg@{repeats}"
    metric_keys = [bottom_pass_key, bottom_key, cons_pass_key, cons_key, mean_key]
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
    total_bottom_tokens = sum(result.cost["bottom_tokens"] for result in doc_results)
    total_repeat_tokens = sum(
        record.full_num_tokens
        for result in doc_results
        for record in result.ranked_repeats
    )
    total_repeat_count = sum(
        len(result.ranked_repeats) for result in doc_results
    )
    total_selected_tokens = sum(
        record.full_num_tokens
        for result in doc_results
        for record in result.ranked_repeats
        if record.selected
    )
    selected_repeat_count = sum(
        1
        for result in doc_results
        for record in result.ranked_repeats
        if record.selected
    )
    saved_tokens = total_full_tokens - total_bottom_tokens
    return {
        "metrics": metrics,
        "cost": {
            "total_full_tokens": total_full_tokens,
            "total_bottom_tokens": total_bottom_tokens,
            "mean_full_tokens_per_doc": total_full_tokens / len(doc_results),
            "mean_bottom_tokens_per_doc": total_bottom_tokens / len(doc_results),
            "mean_full_tokens_per_repeat": total_repeat_tokens / total_repeat_count,
            "mean_selected_tokens_per_selected_repeat": (
                total_selected_tokens / selected_repeat_count
            ),
            "saved_tokens": saved_tokens,
            "saved_pct": saved_tokens / total_full_tokens if total_full_tokens else 0.0,
        },
        "delta": {
            "vs_cons_maj": metrics[bottom_key] - metrics[cons_key],
            "vs_mean_avg": metrics[bottom_key] - metrics[mean_key],
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
    base_cost_definition = build_cost_definition()
    bottom_pass_key = "bottom_pass@1"
    bottom_key = f"bottom_maj@{selected_count}"
    cons_pass_key = "cons_pass@1"
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
        f"{bottom_pass_key}: {summary['metrics'][bottom_pass_key]:.6f}",
        f"{bottom_key}: {summary['metrics'][bottom_key]:.6f}",
        f"{cons_pass_key}: {summary['metrics'][cons_pass_key]:.6f}",
        f"{cons_key}: {summary['metrics'][cons_key]:.6f}",
        f"{mean_key}: {summary['metrics'][mean_key]:.6f}",
        f"delta_vs_cons_maj: {summary['delta']['vs_cons_maj']:.6f}",
        f"delta_vs_mean_avg: {summary['delta']['vs_mean_avg']:.6f}",
        f"total_full_tokens: {summary['cost']['total_full_tokens']}",
        f"total_bottom_tokens: {summary['cost']['total_bottom_tokens']}",
        f"mean_full_tokens_per_doc: {summary['cost']['mean_full_tokens_per_doc']:.6f}",
        f"mean_bottom_tokens_per_doc: {summary['cost']['mean_bottom_tokens_per_doc']:.6f}",
        f"mean_full_tokens_per_repeat: {summary['cost']['mean_full_tokens_per_repeat']:.6f}",
        "mean_selected_tokens_per_selected_repeat: "
        f"{summary['cost']['mean_selected_tokens_per_selected_repeat']:.6f}",
        f"saved_tokens: {summary['cost']['saved_tokens']}",
        f"saved_pct: {summary['cost']['saved_pct']:.6%}",
        f"cost_formula_full_tokens: {base_cost_definition['full_tokens']}",
        "cost_formula_bottom_tokens: "
        "sum_selected(min(prefix_len, full_num_tokens) + full_num_tokens)",
        "cost_formula_mean_full_tokens_per_doc: "
        f"{base_cost_definition['mean_full_tokens_per_doc']}",
        "cost_formula_mean_bottom_tokens_per_doc: total_bottom_tokens / num_docs",
        "cost_formula_mean_full_tokens_per_repeat: "
        f"{base_cost_definition['mean_full_tokens_per_repeat']}",
        "cost_formula_mean_selected_tokens_per_selected_repeat: "
        f"{base_cost_definition['mean_selected_tokens_per_selected_repeat']}",
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
    bottom_fraction: float = 0.5,
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
        bottom_fraction=bottom_fraction,
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
        "cost_definition": {
            **build_cost_definition(),
            "bottom_tokens": (
                "sum_selected(min(prefix_len, full_num_tokens) + full_num_tokens)"
            ),
            "mean_bottom_tokens_per_doc": "total_bottom_tokens / num_docs",
        },
        "summary": summary,
        "docs": [
            {
                "doc_id": result.doc_id,
                "target": result.target,
                "selected_repeat_indices": result.selected_repeat_indices,
                "ranked_repeats": [asdict(record) for record in result.ranked_repeats],
                "selection_stats": asdict(result.selection_stats),
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
        bottom_fraction=args.bottom_fraction,
        selected_count=args.selected_count,
        g=args.g,
        p=args.p,
    )
    print(summary_json)
    print(summary_txt.read_text(encoding="utf-8").rstrip())


if __name__ == "__main__":
    main()
