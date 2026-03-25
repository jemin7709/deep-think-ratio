"""별도 실험 루트와 raw run I/O를 다룬다."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


DEFAULT_SOURCE_ROOT = Path("results")
DEFAULT_TARGET_ROOT = Path("results_deep_think_tokens")
DEFAULT_JSD_DIR_NAME = "deep_think_tokens"


@dataclass(frozen=True)
class Sample:
    """Raw sample row를 replay 가능한 단위로 펼친 결과."""

    doc_id: int
    repeat_index: int
    prompt_text: str
    response_text: str


def format_path_float(value: float) -> str:
    return format(value, "g")


def latest_matching_file(run_dir: Path, pattern: str) -> Path:
    matches = list(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no files matched {pattern} under {run_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def matching_files(run_dir: Path, pattern: str) -> list[Path]:
    return sorted(run_dir.glob(pattern))


def load_aggregated_results(run_dir: Path) -> dict[str, Any]:
    path = latest_matching_file(run_dir, "results_*.json")
    return json.loads(path.read_text(encoding="utf-8"))


def infer_task_name(aggregated: dict[str, Any]) -> str:
    for section in ("results", "configs"):
        keys = list(aggregated.get(section, {}))
        if len(keys) == 1:
            return str(keys[0])
    raise ValueError("could not infer task name from aggregated results")


def latest_samples_file(run_dir: Path, task_name: str) -> Path:
    return latest_matching_file(run_dir, f"samples_{task_name}_*.jsonl")


def resolve_model_path(aggregated: dict[str, Any]) -> str:
    return str(aggregated["config"]["model_args"]["pretrained"])


def _prompt_text_from_arguments(arguments: Any) -> str:
    if isinstance(arguments, dict):
        gen_args = arguments.get("gen_args_0")
        if isinstance(gen_args, dict) and isinstance(gen_args.get("arg_0"), str):
            return str(gen_args["arg_0"])
        first_value = next(iter(arguments.values()), None)
        if isinstance(first_value, dict) and isinstance(first_value.get("arg_0"), str):
            return str(first_value["arg_0"])
    raise ValueError("sample row does not contain a usable prompt in arguments")


def load_samples(
    path: Path,
    repeat_indices: list[int] | None = None,
) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_text = _prompt_text_from_arguments(row["arguments"])
            responses = [str(response) for response in row["resps"][0]]
            indices = (
                list(range(len(responses))) if repeat_indices is None else repeat_indices
            )
            for repeat_index in indices:
                samples.append(
                    Sample(
                        doc_id=int(row["doc_id"]),
                        repeat_index=repeat_index,
                        prompt_text=prompt_text,
                        response_text=responses[repeat_index],
                    )
                )
    return samples


def load_sample_rows(run_dir: Path, task_name: str) -> list[dict[str, Any]]:
    path = latest_samples_file(run_dir, task_name)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tokenize_prompt_and_response(
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
) -> tuple[Tensor, Tensor]:
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_token_ids = tokenizer.encode(response_text, add_special_tokens=False)
    return (
        torch.tensor(prompt_token_ids, dtype=torch.long),
        torch.tensor(response_token_ids, dtype=torch.long),
    )


def copied_run_dir(
    source_run_dir: Path,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    target_root: Path = DEFAULT_TARGET_ROOT,
) -> Path:
    return target_root.resolve() / source_run_dir.resolve().relative_to(
        source_root.resolve()
    )


def discover_run_dirs(source_root: Path = DEFAULT_SOURCE_ROOT) -> list[Path]:
    runs: list[Path] = []
    for result_path in sorted(source_root.resolve().rglob("results_*.json")):
        run_dir = result_path.parent
        if matching_files(run_dir, "samples_*.jsonl"):
            runs.append(run_dir)
    if not runs:
        raise FileNotFoundError(f"no raw run dirs found under {source_root}")
    return runs


def copy_raw_run(
    source_run_dir: Path,
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    target_root: Path = DEFAULT_TARGET_ROOT,
) -> Path:
    target_run_dir = copied_run_dir(
        source_run_dir,
        source_root=source_root,
        target_root=target_root,
    )
    target_run_dir.mkdir(parents=True, exist_ok=True)
    for path in matching_files(source_run_dir, "results_*.json"):
        shutil.copy2(path, target_run_dir / path.name)
    for path in matching_files(source_run_dir, "samples_*.jsonl"):
        shutil.copy2(path, target_run_dir / path.name)
    return target_run_dir


def jsd_output_dir(run_dir: Path) -> Path:
    return run_dir / "jsd_matrices" / DEFAULT_JSD_DIR_NAME


def jsd_matrix_path(output_dir: Path, doc_id: int, repeat_index: int) -> Path:
    return output_dir / f"doc{doc_id}_rep{repeat_index}.pt"


def dtr_results_path(run_dir: Path, *, g: float, p: float) -> Path:
    return (
        run_dir
        / "dtr"
        / f"dtr_g{format_path_float(g)}_p{format_path_float(p)}.json"
    )

