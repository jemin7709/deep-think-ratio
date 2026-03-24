"""현재 lm-eval 결과 디렉터리에서 DTR 계산에 필요한 유틸 모음."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor


HiddenStateMode = Literal["raw_raw", "raw_normed", "normed_normed"]
DEFAULT_HIDDEN_STATE_MODE: HiddenStateMode = "normed_normed"
DEFAULT_TOKEN_BLOCK_SIZE = 128
DEFAULT_G = 0.5
DEFAULT_RHO = 0.85


@dataclass
class Sample:
    """lm-eval 결과에서 DTR 계산에 필요한 최소 단위."""

    doc_id: int
    repeat_index: int
    prompt_text: str
    response_text: str


@dataclass
class PrefillOutput:
    """response 토큰을 예측하는 위치의 hidden state 묶음."""

    response_token_ids: Tensor
    intermediate_hidden_states: list[Tensor]
    final_hidden_states: Tensor


@dataclass
class JSDCacheResult:
    """저장된 JSD matrix에서 복원한 DTR 결과."""

    dtr: float
    settling_depth: Tensor
    deep_mask: Tensor


def latest_matching_file(run_dir: Path, pattern: str) -> Path:
    matches = list(run_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no files matched {pattern} under {run_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_aggregated_results(run_dir: Path) -> dict[str, Any]:
    aggregated_path = latest_matching_file(run_dir, "results_*.json")
    return json.loads(aggregated_path.read_text(encoding="utf-8"))


def infer_task_name(aggregated: dict[str, Any]) -> str:
    for section in ("results", "configs"):
        keys = list(aggregated.get(section, {}))
        if len(keys) == 1:
            return keys[0]
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
        first_value = next(iter(arguments.values()))
        if isinstance(first_value, dict) and isinstance(first_value.get("arg_0"), str):
            return str(first_value["arg_0"])
    raise ValueError("sample row does not contain a usable prompt in arguments")


def load_samples(
    path: Path,
    repeat_indices: list[int] | None = None,
) -> list[Sample]:
    """현재 lm-eval JSONL에서 (doc_id, prompt_text, response_text) 목록을 만든다."""
    samples: list[Sample] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_text = _prompt_text_from_arguments(row["arguments"])
            all_responses = [str(response) for response in row["resps"][0]]
            indices = (
                list(range(len(all_responses)))
                if repeat_indices is None
                else repeat_indices
            )

            for repeat_index in indices:
                samples.append(
                    Sample(
                        doc_id=int(row["doc_id"]),
                        repeat_index=repeat_index,
                        prompt_text=prompt_text,
                        response_text=all_responses[repeat_index],
                    )
                )

    return samples


def tokenize_prompt_and_response(
    tokenizer: Any,
    prompt_text: str,
    response_text: str,
) -> tuple[Tensor, Tensor]:
    """prompt와 response를 각각 토큰화한다."""
    prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    response_token_ids = tokenizer.encode(
        response_text,
        add_special_tokens=False,
    )

    return (
        torch.tensor(prompt_token_ids, dtype=torch.long),
        torch.tensor(response_token_ids, dtype=torch.long),
    )


def _format_path_float(value: float) -> str:
    return format(value, "g")


def jsd_cache_dir_name(
    *,
    hidden_state_mode: HiddenStateMode,
    token_block_size: int,
) -> str:
    return f"{hidden_state_mode}_tb{token_block_size}"


def jsd_output_dir(
    run_dir: Path,
    *,
    hidden_state_mode: HiddenStateMode = DEFAULT_HIDDEN_STATE_MODE,
    token_block_size: int = DEFAULT_TOKEN_BLOCK_SIZE,
) -> Path:
    return (
        run_dir
        / "jsd_matrices"
        / jsd_cache_dir_name(
            hidden_state_mode=hidden_state_mode,
            token_block_size=token_block_size,
        )
    )


def jsd_matrix_path(output_dir: Path, doc_id: int, repeat_index: int) -> Path:
    """샘플 식별자에 대응하는 JSD cache 경로를 만든다."""
    return output_dir / f"doc{doc_id}_rep{repeat_index}.pt"


def dtr_output_dir(run_dir: Path) -> Path:
    """run dir 아래 DTR 관련 산출물을 모을 기본 디렉터리."""
    return run_dir / "dtr"


def dtr_results_path(
    run_dir: Path,
    *,
    g: float = DEFAULT_G,
    rho: float = DEFAULT_RHO,
) -> Path:
    """run dir 기준 기본 DTR JSON 경로."""
    return (
        dtr_output_dir(run_dir)
        / f"dtr_g{_format_path_float(g)}_rho{_format_path_float(rho)}.json"
    )


def validate_dtr_inputs(
    *,
    jsd_matrix: Tensor,
    g: float,
    rho: float,
) -> None:
    num_tokens, num_layers = jsd_matrix.shape
    if num_tokens < 1:
        raise ValueError("jsd_matrix must contain at least one token row")
    if num_layers < 1:
        raise ValueError("jsd_matrix must contain at least one layer column")
    if not 0.0 <= g <= 1.0:
        raise ValueError(f"g must be in the interval [0, 1], got {g}")
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in the interval (0, 1], got {rho}")


def validate_prefix_len(prefix_len: int) -> None:
    if prefix_len < 1:
        raise ValueError(f"prefix_len must be >= 1, got {prefix_len}")


def extract_hidden_states(
    model: Any,
    prompt_token_ids: Tensor,
    response_token_ids: Tensor,
    hidden_state_mode: HiddenStateMode = "raw_raw",
    prefill_chunk_size: int = 512,
    extract_chunk_size: int = 128,
) -> PrefillOutput:
    """KV cache 기반 chunked forward로 response hidden state를 추출한다."""
    device = model.model.embed_tokens.weight.device
    final_norm = model.model.norm
    apply_norm_to_final = hidden_state_mode in ("raw_normed", "normed_normed")
    apply_norm_to_intermediate = hidden_state_mode == "normed_normed"

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
                output_hidden_states=False,
            )
        past_key_values = out.past_key_values
        del out

    tokens_to_extract = torch.cat([prompt_token_ids[-1:], response_token_ids[:-1]])

    all_intermediate_chunks: list[list[Tensor]] = []
    all_final_chunks: list[Tensor] = []
    captured_norm: dict[str, Tensor] = {}

    def _capture_final_norm(module: Any, args: tuple[Tensor, ...], output: Any) -> None:
        del module
        captured_norm["pre"] = args[0][0].detach().cpu()
        post = output if isinstance(output, Tensor) else output[0]
        captured_norm["post"] = post[0].detach().cpu()

    hook = final_norm.register_forward_hook(_capture_final_norm)

    try:
        for chunk_start in range(0, len(tokens_to_extract), extract_chunk_size):
            chunk_end = min(chunk_start + extract_chunk_size, len(tokens_to_extract))
            chunk = tokens_to_extract[chunk_start:chunk_end].unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(
                    chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )

            past_key_values = out.past_key_values
            all_final_chunks.append(
                captured_norm["post" if apply_norm_to_final else "pre"]
            )

            if apply_norm_to_intermediate:
                intermediate_chunk = []
                for hs in out.hidden_states[1:-1]:
                    with torch.no_grad():
                        normed = final_norm(hs)
                    if isinstance(normed, Tensor):
                        intermediate_chunk.append(normed[0].detach().cpu())
                    else:
                        intermediate_chunk.append(normed[0][0].detach().cpu())
                all_intermediate_chunks.append(intermediate_chunk)
            else:
                all_intermediate_chunks.append(
                    [hs[0].detach().cpu() for hs in out.hidden_states[1:-1]]
                )

            del out
    finally:
        hook.remove()

    num_layers = len(all_intermediate_chunks[0])
    intermediate_hidden_states = [
        torch.cat([chunk[i] for chunk in all_intermediate_chunks], dim=0)
        for i in range(num_layers)
    ]
    final_hidden_states = torch.cat(all_final_chunks, dim=0)

    return PrefillOutput(
        response_token_ids=response_token_ids,
        intermediate_hidden_states=intermediate_hidden_states,
        final_hidden_states=final_hidden_states,
    )


def compute_jsd_matrix_from_model(
    model: Any,
    prompt_token_ids: Tensor,
    response_token_ids: Tensor,
    unembed_weight: Tensor,
    hidden_state_mode: HiddenStateMode = "raw_raw",
    prefill_chunk_size: int = 512,
    extract_chunk_size: int = 128,
    token_block_size: int = 128,
) -> Tensor:
    """hidden state를 CPU에 쌓지 않고 extract chunk마다 바로 JSD로 압축한다."""
    device = model.model.embed_tokens.weight.device
    final_norm = model.model.norm
    apply_norm_to_final = hidden_state_mode in ("raw_normed", "normed_normed")
    apply_norm_to_intermediate = hidden_state_mode == "normed_normed"

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
                output_hidden_states=False,
            )
        past_key_values = out.past_key_values
        del out

    tokens_to_extract = torch.cat([prompt_token_ids[-1:], response_token_ids[:-1]])
    all_jsd_chunks: list[Tensor] = []
    captured_norm: dict[str, Tensor] = {}

    def _capture_final_norm(module: Any, args: tuple[Tensor, ...], output: Any) -> None:
        del module
        captured_norm["pre"] = args[0][0].detach()
        post = output if isinstance(output, Tensor) else output[0]
        captured_norm["post"] = post[0].detach()

    hook = final_norm.register_forward_hook(_capture_final_norm)

    try:
        for chunk_start in range(0, len(tokens_to_extract), extract_chunk_size):
            chunk_end = min(chunk_start + extract_chunk_size, len(tokens_to_extract))
            chunk = tokens_to_extract[chunk_start:chunk_end].unsqueeze(0).to(device)
            captured_norm.clear()

            with torch.no_grad():
                out = model(
                    chunk,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                )

            past_key_values = out.past_key_values
            final_hidden_state_chunk = captured_norm[
                "post" if apply_norm_to_final else "pre"
            ]

            if apply_norm_to_intermediate:
                intermediate_hidden_state_chunk = []
                for hs in out.hidden_states[1:-1]:
                    with torch.no_grad():
                        normed = final_norm(hs)
                    if isinstance(normed, Tensor):
                        intermediate_hidden_state_chunk.append(normed[0])
                    else:
                        intermediate_hidden_state_chunk.append(normed[0][0])
            else:
                intermediate_hidden_state_chunk = [
                    hs[0] for hs in out.hidden_states[1:-1]
                ]

            jsd_chunk = compute_jsd_matrix(
                intermediate_hidden_states=intermediate_hidden_state_chunk,
                final_hidden_states=final_hidden_state_chunk,
                unembed_weight=unembed_weight,
                token_block_size=token_block_size,
            )
            all_jsd_chunks.append(jsd_chunk.cpu())

            del out
            del jsd_chunk
    finally:
        hook.remove()

    return torch.cat(all_jsd_chunks, dim=0)


def _jsd_from_logits(logits_final: Tensor, logits_intermediate: Tensor) -> Tensor:
    """두 logits 분포의 JSD를 bits 단위로 계산한다."""
    log_p = F.log_softmax(logits_final.float(), dim=-1)
    log_q = F.log_softmax(logits_intermediate.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()

    mixture = 0.5 * (p + q)
    log_mixture = mixture.clamp(min=1e-12).log()

    kl_pm = (p * (log_p - log_mixture)).sum(dim=-1)
    kl_qm = (q * (log_q - log_mixture)).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm) / math.log(2)


def compute_jsd_matrix(
    intermediate_hidden_states: list[Tensor],
    final_hidden_states: Tensor,
    unembed_weight: Tensor,
    token_block_size: int = 128,
) -> Tensor:
    """hidden state를 token x layer JSD matrix로 압축한다."""
    num_tokens = final_hidden_states.shape[0]
    num_layers = len(intermediate_hidden_states)
    compute_device = unembed_weight.device
    jsd_matrix = torch.zeros(num_tokens, num_layers, device=compute_device)

    for block_start in range(0, num_tokens, token_block_size):
        block_end = min(block_start + token_block_size, num_tokens)
        final_hidden_state_block = final_hidden_states[block_start:block_end].to(
            compute_device
        )
        logits_final = final_hidden_state_block @ unembed_weight.T

        for layer_index, per_layer_hidden_states in enumerate(
            intermediate_hidden_states
        ):
            intermediate_hidden_state_block = per_layer_hidden_states[
                block_start:block_end
            ].to(compute_device)
            logits_intermediate = intermediate_hidden_state_block @ unembed_weight.T
            jsd_matrix[block_start:block_end, layer_index] = _jsd_from_logits(
                logits_final,
                logits_intermediate,
            )

    return jsd_matrix


def compute_dtr_from_jsd_matrix(
    jsd_matrix: Tensor,
    g: float = 0.5,
    rho: float = 0.85,
) -> JSDCacheResult:
    """저장된 JSD matrix만으로 DTR을 계산한다."""
    validate_dtr_inputs(jsd_matrix=jsd_matrix, g=g, rho=rho)
    num_tokens, num_layers = jsd_matrix.shape
    total_layers = num_layers + 1
    deep_start = math.ceil(rho * total_layers)

    cummin_jsd = torch.cummin(jsd_matrix, dim=1).values
    settled_mask = cummin_jsd <= g

    col_idx = torch.arange(num_layers, device=jsd_matrix.device).expand(
        num_tokens,
        -1,
    )
    masked = torch.where(settled_mask, col_idx, num_layers)
    first_settle = masked.min(dim=1).values

    settling_depth = first_settle + 1
    deep_mask = settling_depth >= deep_start
    dtr = deep_mask.float().mean().item()

    return JSDCacheResult(
        dtr=dtr,
        settling_depth=settling_depth,
        deep_mask=deep_mask,
    )
