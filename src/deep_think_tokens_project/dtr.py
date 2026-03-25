"""저장된 divergence cache에서 DTR을 계산한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.deep_think_tokens_project.io import dtr_results_path
from src.deep_think_tokens_project.io import jsd_output_dir
from src.deep_think_tokens_project.utils import DEFAULT_G
from src.deep_think_tokens_project.utils import DEFAULT_P
from src.deep_think_tokens_project.utils import compute_dtr_from_divergence_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate DTR from stored deep-think-tokens divergence caches."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--g", type=float, default=DEFAULT_G)
    parser.add_argument("--p", type=float, default=DEFAULT_P)
    parser.add_argument("--matrix-dir", type=Path)
    parser.add_argument("--output-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_dir = (args.matrix_dir or jsd_output_dir(args.run_dir)).resolve()
    output_path = (
        args.output_path
        or dtr_results_path(args.run_dir.resolve(), g=args.g, p=args.p)
    )
    matrix_paths = sorted(matrix_dir.glob("doc*_rep*.pt"))
    if not matrix_paths:
        raise FileNotFoundError(f"no divergence caches found under {matrix_dir}")

    print(f"처리할 divergence cache 수: {len(matrix_paths)}")
    results = []
    for index, matrix_path in enumerate(matrix_paths, start=1):
        payload = torch.load(matrix_path, map_location="cpu", weights_only=False)
        result = compute_dtr_from_divergence_matrix(
            torch.as_tensor(payload["divergence_matrix"]),
            g=args.g,
            p=args.p,
        )
        entry = {
            "doc_id": int(payload["doc_id"]),
            "repeat_index": int(payload["repeat_index"]),
            "dtr": float(result.dtr),
            "num_tokens": int(payload["num_tokens"]),
            "num_deep_tokens": int(result.deep_mask.sum()),
        }
        results.append(entry)
        print(
            f"[{index}/{len(matrix_paths)}] "
            f"doc={entry['doc_id']} rep={entry['repeat_index']} "
            f"DTR={entry['dtr']:.4f} tokens={entry['num_tokens']} "
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
