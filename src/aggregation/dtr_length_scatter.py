"""run 디렉터리의 DTR와 응답 길이를 묶어 scatter plot 산출물을 만든다."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any

from src.dtr.jsd_utils import infer_task_name
from src.dtr.jsd_utils import (
    dtr_results_path,
    latest_matching_file,
    load_aggregated_results,
    resolve_model_path,
)
from src.plot.dtr_length_scatter import plot_to_png
from tasks.aime24.metrics import TASK_NAME
from tasks.aime24.utils import (
    build_gpt_oss_reasoning_tags,
    resolve_reasoning_tags,
    score_match,
)
from transformers import AutoTokenizer


DEFAULT_OUTPUT_DIR_NAME = "dtr_length_scatter"
OUTPUT_PLOT_FILENAME = "dtr_length_scatter.png"
OUTPUT_JSON_FILENAME = "dtr_length_scatter.json"
OUTPUT_STEM = "dtr_length_scatter"
FULL_LENGTH_MODE = "full"
REASONING_LENGTH_MODE = "reasoning"
DEFAULT_LENGTH_MODE = FULL_LENGTH_MODE
SUPPORTED_LENGTH_MODES = (FULL_LENGTH_MODE, REASONING_LENGTH_MODE)
HARMONY_END_TOKEN = "<|end|>"


@dataclass(frozen=True)
class SequenceLengthPoint:
    """Per-repeat 수준에서 DTR와 응답 길이를 합친 점."""

    doc_id: int
    repeat_index: int
    dtr: float
    response_length: int
    is_correct: bool | None = None


@lru_cache(maxsize=16)
def load_tokenizer(model_name: str) -> Any:
    return AutoTokenizer.from_pretrained(model_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sequence-level DTR vs response length scatter."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dtr-path", type=Path)
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--samples-path", type=Path)
    parser.add_argument("--output-plot", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--length-mode",
        choices=SUPPORTED_LENGTH_MODES,
        default=DEFAULT_LENGTH_MODE,
    )
    parser.add_argument("--title")
    return parser.parse_args()


def build_mode_filename(stem: str, length_mode: str, suffix: str) -> str:
    validate_length_mode(length_mode)
    return f"{stem}.{length_mode}{suffix}"


def build_plot_filename(length_mode: str) -> str:
    return build_mode_filename(OUTPUT_STEM, length_mode, ".png")


def build_summary_filename(length_mode: str) -> str:
    return build_mode_filename(OUTPUT_STEM, length_mode, ".json")


def validate_length_mode(length_mode: str) -> str:
    if length_mode not in SUPPORTED_LENGTH_MODES:
        raise ValueError(
            f"unsupported length_mode {length_mode!r}; expected one of {SUPPORTED_LENGTH_MODES}"
        )
    return length_mode


def build_length_axis_label(length_mode: str) -> str:
    return (
        "Reasoning Length (tokens)"
        if validate_length_mode(length_mode) == REASONING_LENGTH_MODE
        else "Response Length (tokens)"
    )


def _correctness_length_stats(
    points: list[SequenceLengthPoint],
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


def _build_title_suffix(length_mode: str) -> str:
    return (
        "DTR vs Reasoning Length"
        if validate_length_mode(length_mode) == REASONING_LENGTH_MODE
        else "DTR vs Response Length"
    )


def resolve_output_plot_path(output_plot: Path | None, *, length_mode: str) -> Path | None:
    if output_plot is None:
        return None

    candidate = output_plot.resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / build_plot_filename(length_mode)
    if candidate.suffix:
        return candidate
    return candidate / build_plot_filename(length_mode)


def resolve_output_json_path(output_json: Path | None, *, length_mode: str) -> Path:
    if output_json is None:
        raise ValueError("output_json must not be None")

    candidate = output_json.resolve()
    if candidate.exists() and candidate.is_dir():
        return candidate / build_summary_filename(length_mode)
    if candidate.suffix:
        return candidate
    return candidate / build_summary_filename(length_mode)


def default_output_dir(run_dir: Path) -> Path:
    """개별 run의 DTR-length scatter 산출물을 모으는 기본 디렉터리."""
    return run_dir / DEFAULT_OUTPUT_DIR_NAME


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None, Path]:
    run_dir = args.run_dir.resolve()
    length_mode = validate_length_mode(getattr(args, "length_mode", DEFAULT_LENGTH_MODE))
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
        resolve_output_plot_path(args.output_plot, length_mode=length_mode)
        if args.output_plot is not None
        else output_dir / build_plot_filename(length_mode)
    )
    output_json = (
        resolve_output_json_path(args.output_json, length_mode=length_mode)
        if args.output_json is not None
        else output_dir / build_summary_filename(length_mode)
    )
    return dtr_path, results_path, output_plot, output_json


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

    # Prefer the sample snapshot that shares the same results suffix so we
    # never silently recolor points from a different run snapshot.
    suffix = _results_suffix(results_path)
    if suffix is not None:
        matched = run_dir / f"samples_{task_name}_{suffix}.jsonl"
        if matched.is_file():
            return matched

    candidates = sorted(run_dir.glob(f"samples_{task_name}_*.jsonl"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"no samples_{task_name}_*.jsonl found under {run_dir}")
    raise FileNotFoundError(
        "multiple sample files found without an exact match for "
        f"{results_path.name}; pass --samples-path explicitly"
    )


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
                is_correct=(
                    None
                    if row.get("is_correct") is None
                    else bool(row["is_correct"])
                ),
            )
        )

    points.sort(key=lambda point: (point.dtr, point.doc_id, point.repeat_index))
    return points


def resolve_length_reasoning_tags(
    aggregated: dict[str, Any],
    *,
    length_mode: str,
) -> list[tuple[str, str]] | None:
    validate_length_mode(length_mode)
    reasoning_tags = resolve_reasoning_tags(aggregated)
    if reasoning_tags is not None or length_mode != REASONING_LENGTH_MODE:
        return reasoning_tags

    model_path = resolve_model_path(aggregated)
    if "gpt-oss" in model_path:
        return build_gpt_oss_reasoning_tags()
    return None


def _validate_harmony_reasoning_tags(
    reasoning_tags: list[tuple[str, str]] | None,
) -> tuple[str, str]:
    expected = build_gpt_oss_reasoning_tags()
    if reasoning_tags != expected:
        raise ValueError(
            "reasoning length mode requires GPT-OSS harmony reasoning tags"
        )
    return expected[0]


def extract_harmony_reasoning_text(
    completion: str,
    *,
    reasoning_tags: list[tuple[str, str]] | None,
) -> str:
    start_tag, end_tag = _validate_harmony_reasoning_tags(reasoning_tags)
    start_index = completion.find(start_tag)
    if start_index < 0:
        raise ValueError("missing harmony analysis start tag in completion")
    content_start = start_index + len(start_tag)
    end_index = completion.find(end_tag, content_start)
    if end_index < 0:
        raise ValueError("missing harmony reasoning end boundary in completion")
    if not end_tag.startswith(HARMONY_END_TOKEN):
        raise ValueError("unsupported harmony reasoning end tag")
    return completion[content_start:end_index] + HARMONY_END_TOKEN


def count_response_tokens(text: str, *, model_name: str) -> int:
    tokenizer = load_tokenizer(model_name)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return int(len(token_ids))


def load_response_lengths_by_key(
    samples_path: Path,
    *,
    length_mode: str,
    model_name: str,
    reasoning_tags: list[tuple[str, str]] | None,
) -> dict[tuple[int, int], int]:
    validate_length_mode(length_mode)
    length_by_key: dict[tuple[int, int], int] = {}

    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            doc_id = int(sample["doc_id"])
            responses = [str(response) for response in sample["resps"][0]]

            for repeat_index, response in enumerate(responses):
                measured = (
                    extract_harmony_reasoning_text(
                        response,
                        reasoning_tags=reasoning_tags,
                    )
                    if length_mode == REASONING_LENGTH_MODE
                    else response
                )
                length_by_key[(doc_id, repeat_index)] = count_response_tokens(
                    measured,
                    model_name=model_name,
                )

    return length_by_key


def load_correctness_by_key(
    samples_path: Path,
    *,
    reasoning_tags: list[tuple[str, str]] | None,
) -> dict[tuple[int, int], bool]:
    correctness_by_key: dict[tuple[int, int], bool] = {}

    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            doc_id = int(sample["doc_id"])
            target = str(sample["target"])
            responses = [str(response) for response in sample["resps"][0]]

            for repeat_index, response in enumerate(responses):
                correctness_by_key[(doc_id, repeat_index)] = (
                    score_match(
                        target,
                        response,
                        reasoning_tags=reasoning_tags,
                    )
                    == 1.0
                )

    return correctness_by_key


def attach_correctness(
    points: list[SequenceLengthPoint],
    correctness_by_key: dict[tuple[int, int], bool],
) -> list[SequenceLengthPoint]:
    return [
        SequenceLengthPoint(
            doc_id=point.doc_id,
            repeat_index=point.repeat_index,
            dtr=point.dtr,
            response_length=point.response_length,
            is_correct=correctness_by_key[(point.doc_id, point.repeat_index)],
        )
        for point in points
    ]


def attach_response_lengths(
    points: list[SequenceLengthPoint],
    response_lengths: dict[tuple[int, int], int],
) -> list[SequenceLengthPoint]:
    return [
        SequenceLengthPoint(
            doc_id=point.doc_id,
            repeat_index=point.repeat_index,
            dtr=point.dtr,
            response_length=response_lengths[(point.doc_id, point.repeat_index)],
            is_correct=point.is_correct,
        )
        for point in points
    ]


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
    length_mode: str = DEFAULT_LENGTH_MODE,
) -> str:
    if user_title is not None:
        return user_title
    return (
        f"{run_dir.name} | {task_name} | {model_name} | "
        f"{_build_title_suffix(length_mode)}"
    )


def write_summary_json(
    *,
    run_dir: Path,
    task_name: str,
    model_name: str,
    dtr_path: Path,
    results_path: Path,
    samples_path: Path | None,
    output_path: Path,
    points: list[SequenceLengthPoint],
    pearson: float,
    length_mode: str,
) -> None:
    dtrs = [point.dtr for point in points]
    lengths = [point.response_length for point in points]
    summary = {
        "run_dir": str(run_dir),
        "task": task_name,
        "model": model_name,
        "length_mode": validate_length_mode(length_mode),
        "dtr_path": str(dtr_path),
        "results_path": str(results_path),
        "samples_path": None if samples_path is None else str(samples_path),
        "num_sequences": len(points),
        "pearson_r": pearson,
        "dtr_min": min(dtrs),
        "dtr_max": max(dtrs),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": fmean(lengths),
        **_correctness_length_stats(points),
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
    length_mode = validate_length_mode(args.length_mode)
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
    samples_path = None
    if task_name == TASK_NAME:
        samples_path = resolve_samples_path(
            run_dir=run_dir,
            task_name=task_name,
            results_path=results_path,
            samples_path=getattr(args, "samples_path", None),
        )
        reasoning_tags = resolve_length_reasoning_tags(
            aggregated,
            length_mode=length_mode,
        )
        if length_mode == REASONING_LENGTH_MODE:
            points = attach_response_lengths(
                points,
                load_response_lengths_by_key(
                    samples_path,
                    length_mode=length_mode,
                    model_name=resolve_model_path(aggregated),
                    reasoning_tags=reasoning_tags,
                ),
            )
        points = attach_correctness(
            points,
            load_correctness_by_key(
                samples_path,
                reasoning_tags=reasoning_tags,
            ),
        )
    pearson = pearson_r(
        [point.dtr for point in points],
        [float(point.response_length) for point in points],
    )

    if output_plot is not None:
        plot_to_png(
            points=points,
            pearson=pearson,
            output_path=output_plot,
            title=build_title(
                run_dir,
                task_name,
                model_name,
                args.title,
                length_mode=length_mode,
            ),
            y_label=build_length_axis_label(length_mode),
        )
    write_summary_json(
        run_dir=run_dir,
        task_name=task_name,
        model_name=model_name,
        dtr_path=dtr_path,
        results_path=results_path,
        samples_path=samples_path,
        output_path=output_json,
        points=points,
        pearson=pearson,
        length_mode=length_mode,
    )
    print_summary(points, pearson)
    print(f"Saved summary: {output_json}")
    if output_plot is not None:
        print(f"Saved plot: {output_plot}")


if __name__ == "__main__":
    main()
