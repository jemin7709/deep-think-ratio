"""Replay 기반 divergence cache를 생성한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.deep_think_tokens_project.hooks import add_deep_thinking_tokens_hooks
from src.deep_think_tokens_project.io import infer_task_name
from src.deep_think_tokens_project.io import jsd_matrix_path
from src.deep_think_tokens_project.io import jsd_output_dir
from src.deep_think_tokens_project.io import latest_samples_file
from src.deep_think_tokens_project.io import load_aggregated_results
from src.deep_think_tokens_project.io import load_samples
from src.deep_think_tokens_project.io import resolve_model_path
from src.deep_think_tokens_project.io import tokenize_prompt_and_response
from src.deep_think_tokens_project.utils import div_to_matrix


DEFAULT_PREFILL_CHUNK_SIZE = 4096
DEFAULT_EXTRACT_CHUNK_SIZE = 1024


def replay_tokens(
    prompt_token_ids: torch.Tensor,
    response_token_ids: torch.Tensor,
) -> torch.Tensor:
    if prompt_token_ids.numel() < 1:
        raise ValueError("prompt_token_ids must contain at least one token")
    if response_token_ids.numel() < 1:
        raise ValueError("response_token_ids must contain at least one token")
    return torch.cat([prompt_token_ids[-1:], response_token_ids[:-1]])


def replay_response_divergences(
    model,
    prompt_token_ids: torch.Tensor,
    response_token_ids: torch.Tensor,
    *,
    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE,
    extract_chunk_size: int = DEFAULT_EXTRACT_CHUNK_SIZE,
) -> torch.Tensor:
    device = next(model.parameters()).device
    past_key_values = None
    prompt_prefix = prompt_token_ids[:-1]

    for chunk_start in range(0, len(prompt_prefix), prefill_chunk_size):
        chunk_end = min(chunk_start + prefill_chunk_size, len(prompt_prefix))
        chunk = prompt_prefix[chunk_start:chunk_end].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(
                chunk,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = out.past_key_values
        del out

    tracker = add_deep_thinking_tokens_hooks(model, clear_on_generate=False)
    tokens_to_extract = replay_tokens(prompt_token_ids, response_token_ids)

    try:
        for chunk_start in range(0, len(tokens_to_extract), extract_chunk_size):
            chunk_end = min(chunk_start + extract_chunk_size, len(tokens_to_extract))
            chunk = tokens_to_extract[chunk_start:chunk_end].unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(
                    chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            del out
        divergences = tracker.collect()
    finally:
        tracker.detach()

    return div_to_matrix(divergences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deep-think-tokens divergence caches from replay."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--repeat-indices", type=int, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--prefill-chunk-size", type=int, default=DEFAULT_PREFILL_CHUNK_SIZE
    )
    parser.add_argument(
        "--extract-chunk-size", type=int, default=DEFAULT_EXTRACT_CHUNK_SIZE
    )
    return parser.parse_args()


def build_jsd_payload(
    *,
    doc_id: int,
    repeat_index: int,
    model_path: str,
    task_name: str,
    samples_path: Path,
    response_token_ids: torch.Tensor,
    divergence_matrix: torch.Tensor,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "repeat_index": repeat_index,
        "model_path": model_path,
        "task_name": task_name,
        "samples_path": str(samples_path),
        "num_tokens": int(response_token_ids.shape[0]),
        "response_token_ids": response_token_ids.cpu(),
        "divergence_matrix": divergence_matrix.cpu(),
    }


def _payload_identity(payload: dict[str, object]) -> dict[str, object]:
    return {
        "doc_id": int(payload["doc_id"]),  # type: ignore[arg-type]
        "repeat_index": int(payload["repeat_index"]),  # type: ignore[arg-type]
        "model_path": str(payload["model_path"]),
        "task_name": str(payload["task_name"]),
        "samples_path": str(payload["samples_path"]),
    }


def save_jsd_payload(output_path: Path, payload: dict[str, object]) -> None:
    if output_path.exists():
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        if _payload_identity(existing) != _payload_identity(payload):
            raise FileExistsError(
                "refusing to overwrite divergence cache with mismatched metadata: "
                f"{output_path}"
            )
    torch.save(payload, output_path)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or jsd_output_dir(run_dir)).resolve()
    aggregated = load_aggregated_results(run_dir)
    task_name = args.task or infer_task_name(aggregated)
    samples_path = latest_samples_file(run_dir, task_name)
    model_path = resolve_model_path(aggregated)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    samples = load_samples(samples_path, args.repeat_indices)

    print(f"run dir: {run_dir}")
    print(f"샘플 파일: {samples_path}")
    print(f"모델: {model_path}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"처리할 샘플 수: {len(samples)}")

    manifest_entries: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        prompt_token_ids, response_token_ids = tokenize_prompt_and_response(
            tokenizer,
            sample.prompt_text,
            sample.response_text,
        )
        divergence_matrix = replay_response_divergences(
            model,
            prompt_token_ids,
            response_token_ids,
            prefill_chunk_size=args.prefill_chunk_size,
            extract_chunk_size=args.extract_chunk_size,
        )
        output_path = jsd_matrix_path(output_dir, sample.doc_id, sample.repeat_index)
        payload = build_jsd_payload(
            doc_id=sample.doc_id,
            repeat_index=sample.repeat_index,
            model_path=model_path,
            task_name=task_name,
            samples_path=samples_path,
            response_token_ids=response_token_ids,
            divergence_matrix=divergence_matrix,
        )
        save_jsd_payload(output_path, payload)
        manifest_entries.append(
            {
                "doc_id": sample.doc_id,
                "repeat_index": sample.repeat_index,
                "num_layers": int(divergence_matrix.shape[0]),
                "num_tokens": int(divergence_matrix.shape[1]),
                "path": str(output_path),
            }
        )
        print(
            f"[{index}/{len(samples)}] "
            f"doc={sample.doc_id} rep={sample.repeat_index} "
            f"layers={divergence_matrix.shape[0]} tokens={divergence_matrix.shape[1]} "
            f"saved={output_path.name}"
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "task_name": task_name,
                "samples_path": str(samples_path),
                "model_path": model_path,
                "num_samples": len(manifest_entries),
                "files": manifest_entries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"manifest 저장: {manifest_path}")


if __name__ == "__main__":
    main()
