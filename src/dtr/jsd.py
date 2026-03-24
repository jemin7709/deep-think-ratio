"""현재 results run dir에서 JSD matrix 캐시를 생성한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import torch

from src.plot.jsd_heatmap import render_heatmap
from src.dtr.jsd_utils import DEFAULT_HIDDEN_STATE_MODE
from src.dtr.jsd_utils import DEFAULT_TOKEN_BLOCK_SIZE
from src.dtr.jsd_utils import HiddenStateMode
from src.dtr.jsd_utils import compute_jsd_matrix_from_model
from src.dtr.jsd_utils import infer_task_name
from src.dtr.jsd_utils import jsd_matrix_path
from src.dtr.jsd_utils import jsd_output_dir
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
    parser.add_argument("--token-block-size", type=int, default=DEFAULT_TOKEN_BLOCK_SIZE)
    parser.add_argument(
        "--hidden-state-mode",
        type=str,
        choices=["raw_raw", "raw_normed", "normed_normed"],
        default=DEFAULT_HIDDEN_STATE_MODE,
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
        help=(
            "JSD matrix 저장 디렉토리 "
            "(기본값: <run_dir>/jsd_matrices/<hidden_state_mode>_tb<token_block_size>)"
        ),
    )
    parser.add_argument(
        "--save-heatmap",
        action="store_true",
        help="각 JSD matrix에 대응하는 heatmap PNG도 함께 저장",
    )
    parser.add_argument(
        "--render-heatmaps-only",
        action="store_true",
        help="기존 JSD cache 디렉토리의 *.pt만 읽어 heatmap PNG만 다시 생성",
    )
    parser.add_argument(
        "--heatmap-dir",
        type=Path,
        default=None,
        help="heatmap 저장 디렉토리 (기본값: <output_dir>/heatmaps)",
    )
    parser.add_argument(
        "--heatmap-cell-width",
        type=int,
        default=None,
        help="heatmap 토큰 축 셀 너비",
    )
    parser.add_argument(
        "--heatmap-cell-height",
        type=int,
        default=None,
        help="heatmap 레이어 축 셀 높이",
    )
    parser.add_argument(
        "--max-token-labels",
        type=int,
        default=None,
        help="heatmap X축에 표시할 최대 토큰 라벨 수",
    )
    parser.add_argument(
        "--max-layer-labels",
        type=int,
        default=24,
        help="heatmap Y축에 표시할 최대 레이어 라벨 수",
    )
    parser.add_argument(
        "--heatmap-vmax",
        type=float,
        default=None,
        help="heatmap color scale 상한. 생략하면 matrix 최댓값 사용",
    )
    parser.add_argument(
        "--heatmap-font-size",
        type=int,
        default=14,
        help="heatmap 축 라벨 폰트 크기",
    )
    return parser.parse_args()


def escape_token_label(text: str) -> str:
    if not text:
        return "<empty>"
    return text.replace("\n", "\\n").replace("\t", "\\t")


def build_token_labels(tokenizer: Any, response_token_ids: torch.Tensor) -> list[str]:
    labels: list[str] = []
    for token_id in response_token_ids.tolist():
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            decoded = tokenizer.convert_ids_to_tokens(
                [token_id],
                skip_special_tokens=False,
            )[0]
        labels.append(escape_token_label(decoded))
    return labels


def build_heatmap_title(
    *,
    doc_id: int,
    repeat_index: int,
    hidden_state_mode: HiddenStateMode,
) -> str:
    return f"doc={doc_id} rep={repeat_index} mode={hidden_state_mode}"


def build_heatmap_subtitle(
    *,
    num_tokens: int,
    num_layers: int,
) -> str:
    return f"{num_tokens} tokens x {num_layers} layers (top=last, bottom=0)"


def heatmap_path(heatmap_dir: Path, doc_id: int, repeat_index: int) -> Path:
    return heatmap_dir / f"doc{doc_id}_rep{repeat_index}.png"


def resolve_hidden_state_mode(raw: object) -> HiddenStateMode:
    if raw in ("raw_raw", "raw_normed", "normed_normed"):
        return raw
    raise ValueError(f"unsupported hidden_state_mode in cached payload: {raw}")


def resolve_heatmap_dir(output_dir: Path, heatmap_dir: Path | None) -> Path:
    return (heatmap_dir or output_dir / "heatmaps").resolve()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return jsd_output_dir(
        args.run_dir.resolve(),
        hidden_state_mode=args.hidden_state_mode,
        token_block_size=args.token_block_size,
    ).resolve()


def build_jsd_payload(
    *,
    doc_id: int,
    repeat_index: int,
    model_path: str,
    task_name: str,
    samples_path: Path,
    hidden_state_mode: HiddenStateMode,
    token_block_size: int,
    response_token_ids: torch.Tensor,
    jsd_matrix: torch.Tensor,
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "repeat_index": repeat_index,
        "model_path": model_path,
        "task_name": task_name,
        "samples_path": str(samples_path),
        "hidden_state_mode": hidden_state_mode,
        "token_block_size": token_block_size,
        "num_tokens": int(response_token_ids.shape[0]),
        "response_token_ids": response_token_ids.cpu(),
        "jsd_matrix": jsd_matrix,
    }


def _payload_identity(payload: dict[str, object]) -> dict[str, object]:
    return {
        "hidden_state_mode": payload["hidden_state_mode"],
        "token_block_size": cast(int, payload["token_block_size"]),
        "model_path": str(payload["model_path"]),
        "task_name": str(payload["task_name"]),
        "samples_path": str(payload["samples_path"]),
    }


def save_jsd_payload(output_path: Path, payload: dict[str, object]) -> None:
    if output_path.exists():
        existing = torch.load(output_path, map_location="cpu", weights_only=False)
        if _payload_identity(existing) != _payload_identity(payload):
            raise FileExistsError(
                f"refusing to overwrite JSD cache with mismatched metadata: {output_path}"
            )
    torch.save(payload, output_path)


def render_existing_heatmaps(
    *,
    output_dir: Path,
    heatmap_dir: Path,
    heatmap_cell_width: int | None,
    heatmap_cell_height: int | None,
    max_token_labels: int | None,
    max_layer_labels: int,
    heatmap_font_size: int,
    heatmap_vmax: float | None,
) -> None:
    matrix_paths = sorted(output_dir.glob("doc*_rep*.pt"))
    if not matrix_paths:
        raise FileNotFoundError(f"no JSD matrices found under {output_dir}")

    from transformers import AutoTokenizer

    tokenizer: Any | None = None
    tokenizer_model_path: str | None = None
    print(f"heatmap-only 입력 디렉토리: {output_dir}")
    print(f"heatmap 출력 디렉토리: {heatmap_dir}")
    print(f"처리할 JSD matrix 수: {len(matrix_paths)}")

    for index, matrix_path in enumerate(matrix_paths, start=1):
        payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
        model_path = str(payload["model_path"])
        if tokenizer is None or tokenizer_model_path != model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer_model_path = model_path

        doc_id = int(payload["doc_id"])
        repeat_index = int(payload["repeat_index"])
        response_token_ids = torch.as_tensor(payload["response_token_ids"], dtype=torch.long)
        jsd_matrix = torch.as_tensor(payload["jsd_matrix"])
        hidden_state_mode = resolve_hidden_state_mode(
            payload.get("hidden_state_mode", "normed_normed")
        )

        plot_path = heatmap_path(heatmap_dir, doc_id, repeat_index)
        token_labels = build_token_labels(tokenizer, response_token_ids)
        render_heatmap(
            jsd_matrix=jsd_matrix,
            token_labels=token_labels,
            title=build_heatmap_title(
                doc_id=doc_id,
                repeat_index=repeat_index,
                hidden_state_mode=hidden_state_mode,
            ),
            subtitle=build_heatmap_subtitle(
                num_tokens=int(response_token_ids.shape[0]),
                num_layers=int(jsd_matrix.shape[1]),
            ),
            output_path=plot_path,
            cell_width=heatmap_cell_width,
            cell_height=heatmap_cell_height,
            max_token_labels=max_token_labels,
            max_layer_labels=max_layer_labels,
            font_size=heatmap_font_size,
            vmax=heatmap_vmax,
        )
        print(
            f"[{index}/{len(matrix_paths)}] "
            f"doc={doc_id} rep={repeat_index} "
            f"plot={plot_path.name}"
        )


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = resolve_output_dir(args)
    should_render_heatmaps = args.save_heatmap or args.render_heatmaps_only
    if args.render_heatmaps_only:
        heatmap_dir = resolve_heatmap_dir(output_dir, args.heatmap_dir)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        render_existing_heatmaps(
            output_dir=output_dir,
            heatmap_dir=heatmap_dir,
            heatmap_cell_width=args.heatmap_cell_width,
            heatmap_cell_height=args.heatmap_cell_height,
            max_token_labels=args.max_token_labels,
            max_layer_labels=args.max_layer_labels,
            heatmap_font_size=args.heatmap_font_size,
            heatmap_vmax=args.heatmap_vmax,
        )
        return

    aggregated = load_aggregated_results(run_dir)
    task_name = args.task or infer_task_name(aggregated)
    samples_path = latest_samples_file(run_dir, task_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = (
        resolve_heatmap_dir(output_dir, args.heatmap_dir)
        if should_render_heatmaps
        else None
    )
    if heatmap_dir is not None:
        heatmap_dir.mkdir(parents=True, exist_ok=True)

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
    if heatmap_dir is not None:
        print(f"heatmap 출력 디렉토리: {heatmap_dir}")

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
        payload = build_jsd_payload(
            doc_id=sample.doc_id,
            repeat_index=sample.repeat_index,
            model_path=model_path,
            task_name=task_name,
            samples_path=samples_path,
            hidden_state_mode=hidden_state_mode,
            token_block_size=args.token_block_size,
            response_token_ids=response_token_ids,
            jsd_matrix=jsd_matrix,
        )
        save_jsd_payload(output_path, payload)
        plot_path: Path | None = None
        if heatmap_dir is not None:
            plot_path = heatmap_path(heatmap_dir, sample.doc_id, sample.repeat_index)
            token_labels = build_token_labels(tokenizer, response_token_ids.cpu())
            render_heatmap(
                jsd_matrix=jsd_matrix,
                token_labels=token_labels,
                title=build_heatmap_title(
                    doc_id=sample.doc_id,
                    repeat_index=sample.repeat_index,
                    hidden_state_mode=hidden_state_mode,
                ),
                subtitle=build_heatmap_subtitle(
                    num_tokens=int(response_token_ids.shape[0]),
                    num_layers=int(jsd_matrix.shape[1]),
                ),
                output_path=plot_path,
                cell_width=args.heatmap_cell_width,
                cell_height=args.heatmap_cell_height,
                max_token_labels=args.max_token_labels,
                max_layer_labels=args.max_layer_labels,
                font_size=args.heatmap_font_size,
                vmax=args.heatmap_vmax,
            )

        manifest_entries.append(
            {
                "doc_id": sample.doc_id,
                "repeat_index": sample.repeat_index,
                "num_tokens": int(response_token_ids.shape[0]),
                "path": str(output_path),
                "plot_path": str(plot_path) if plot_path is not None else "",
            }
        )

        saved_plot = f" plot={plot_path.name}" if plot_path is not None else ""
        print(
            f"[{index}/{len(samples)}] "
            f"doc={sample.doc_id} rep={sample.repeat_index} "
            f"tokens={response_token_ids.shape[0]} "
            f"saved={output_path.name}"
            f"{saved_plot}"
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
                "heatmap_dir": str(heatmap_dir) if heatmap_dir is not None else "",
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
