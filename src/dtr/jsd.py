"""현재 results run dir에서 JSD matrix 캐시를 생성한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.dtr.jsd_utils import HiddenStateMode
from src.dtr.jsd_utils import compute_jsd_matrix_from_model
from src.dtr.jsd_utils import infer_task_name
from src.dtr.jsd_utils import jsd_matrix_path
from src.dtr.jsd_utils import latest_samples_file
from src.dtr.jsd_utils import load_aggregated_results
from src.dtr.jsd_utils import load_samples
from src.dtr.jsd_utils import resolve_model_path
from src.dtr.jsd_utils import tokenize_prompt_and_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run dir 기준 JSD matrix 캐시 생성")
    parser.add_argument("run_dir", type=Path, help="results/<task>/<model>/<seed>/<timestamp>")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="샘플 파일명을 찾을 task 이름 (기본값: aggregated results에서 추론)",
    )
    parser.add_argument(
        "--repeat-indices",
        type=int,
        nargs="*",
        default=None,
        help="추출할 repeat index 목록 (생략 시 전체)",
    )
    parser.add_argument("--token-block-size", type=int, default=128)
    parser.add_argument(
        "--hidden-state-mode",
        type=str,
        choices=["raw_raw", "raw_normed", "normed_normed"],
        default="normed_normed",
        help=(
            "hidden state 추출 모드: "
            "raw_raw=중간 raw/마지막 raw, "
            "raw_normed=중간 raw/마지막 normed, "
            "normed_normed=중간 normed/마지막 normed"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="JSD matrix 저장 디렉토리 (기본값: <run_dir>/jsd_matrices)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    aggregated = load_aggregated_results(run_dir)
    task_name = args.task or infer_task_name(aggregated)
    samples_path = latest_samples_file(run_dir, task_name)
    output_dir = (args.output_dir or run_dir / "jsd_matrices").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_model_path(aggregated)
    hidden_state_mode: HiddenStateMode = args.hidden_state_mode

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    ).eval()

    unembed_weight = model.lm_head.weight.detach()
    samples = load_samples(samples_path, args.repeat_indices)
    print(f"처리할 샘플 수: {len(samples)}")
    print(f"run dir: {run_dir}")
    print(f"샘플 파일: {samples_path}")
    print(f"JSD matrix 출력 디렉토리: {output_dir}")
    print(f"hidden state 모드: {hidden_state_mode}")

    manifest_entries: list[dict[str, int | str]] = []

    for index, sample in enumerate(samples, start=1):
        prompt_token_ids, response_token_ids = tokenize_prompt_and_response(
            tokenizer,
            sample.prompt_text,
            sample.response_text,
        )

        jsd_matrix = compute_jsd_matrix_from_model(
            model=model,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            unembed_weight=unembed_weight,
            hidden_state_mode=hidden_state_mode,
            token_block_size=args.token_block_size,
        )

        output_path = jsd_matrix_path(
            output_dir,
            sample.doc_id,
            sample.repeat_index,
        )
        torch.save(
            {
                "doc_id": sample.doc_id,
                "repeat_index": sample.repeat_index,
                "model_path": model_path,
                "task_name": task_name,
                "samples_path": str(samples_path),
                "hidden_state_mode": hidden_state_mode,
                "token_block_size": args.token_block_size,
                "num_tokens": int(response_token_ids.shape[0]),
                "response_token_ids": response_token_ids.cpu(),
                "jsd_matrix": jsd_matrix,
            },
            output_path,
        )

        manifest_entries.append(
            {
                "doc_id": sample.doc_id,
                "repeat_index": sample.repeat_index,
                "num_tokens": int(response_token_ids.shape[0]),
                "path": str(output_path),
            }
        )

        print(
            f"[{index}/{len(samples)}] "
            f"doc={sample.doc_id} rep={sample.repeat_index} "
            f"tokens={response_token_ids.shape[0]} "
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
                "hidden_state_mode": hidden_state_mode,
                "token_block_size": args.token_block_size,
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
