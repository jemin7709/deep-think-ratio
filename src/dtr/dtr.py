"""저장된 JSD matrix만 읽어 현재 run dir 기준 DTR을 다시 계산한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.dtr.jsd_utils import compute_dtr_from_jsd_matrix
from src.dtr.jsd_utils import DEFAULT_G
from src.dtr.jsd_utils import DEFAULT_HIDDEN_STATE_MODE
from src.dtr.jsd_utils import DEFAULT_RHO
from src.dtr.jsd_utils import DEFAULT_TOKEN_BLOCK_SIZE
from src.dtr.jsd_utils import dtr_results_path
from src.dtr.jsd_utils import jsd_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="저장된 JSD matrix로 DTR 계산")
    parser.add_argument(
        "run_dir",
        type=Path,
        help="결과 run 디렉토리 경로 (e.g. results/aime24_custom/model/0/<timestamp>)",
    )
    parser.add_argument("--g", type=float, default=DEFAULT_G, help="settling threshold")
    parser.add_argument(
        "--rho",
        type=float,
        default=DEFAULT_RHO,
        help="deep-thinking depth fraction",
    )
    parser.add_argument(
        "--hidden-state-mode",
        type=str,
        choices=["raw_raw", "raw_normed", "normed_normed"],
        default=DEFAULT_HIDDEN_STATE_MODE,
        help="기본 matrix-dir를 해석할 hidden state 모드",
    )
    parser.add_argument(
        "--token-block-size",
        type=int,
        default=DEFAULT_TOKEN_BLOCK_SIZE,
        help="기본 matrix-dir를 해석할 token block size",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        help=(
            "JSD matrix 디렉토리 "
            "(기본값: <run_dir>/jsd_matrices/<hidden_state_mode>_tb<token_block_size>)"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="결과 JSON 경로 (기본값: <run_dir>/dtr/dtr_g{g}_rho{rho}.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    matrix_dir = args.matrix_dir or jsd_output_dir(
        args.run_dir,
        hidden_state_mode=args.hidden_state_mode,
        token_block_size=args.token_block_size,
    )
    output_path = args.output_path or dtr_results_path(args.run_dir, g=args.g, rho=args.rho)

    matrix_paths = sorted(matrix_dir.glob("doc*_rep*.pt"))
    if not matrix_paths:
        raise FileNotFoundError(f"no JSD matrices found under {matrix_dir}")
    print(f"처리할 JSD matrix 수: {len(matrix_paths)}")

    results = []

    for index, matrix_path in enumerate(matrix_paths, start=1):
        payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
        dtr_result = compute_dtr_from_jsd_matrix(
            payload["jsd_matrix"],
            g=args.g,
            rho=args.rho,
        )

        entry = {
            "doc_id": int(payload["doc_id"]),
            "repeat_index": int(payload["repeat_index"]),
            "dtr": float(dtr_result.dtr),
            "num_tokens": int(payload["num_tokens"]),
            "num_deep_tokens": int(dtr_result.deep_mask.sum()),
        }
        results.append(entry)

        print(
            f"[{index}/{len(matrix_paths)}] "
            f"doc={entry['doc_id']} rep={entry['repeat_index']} "
            f"DTR={entry['dtr']:.4f} "
            f"tokens={entry['num_tokens']} "
            f"deep={entry['num_deep_tokens']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"결과 저장: {output_path}")


if __name__ == "__main__":
    main()
