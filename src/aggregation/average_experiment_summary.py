"""같은 experiment slug를 가진 `summary.json` 산출물을 run들에 걸쳐 평균낸다.

`src.experiment.think_n`은 각 run 아래 `experiments/<slug>/summary.json`을 만든다.
이 모듈은 예를 들어 `prefix50_top50` 같은 slug를 받아, 해당 slug의 summary만 모아
metric, cost, delta 섹션의 평균과 표준편차를 저장한다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


DEFAULT_INPUT_ROOT = Path("results")
DEFAULT_AGGREGATE_DIR_NAME = "experiments_aggregated"
SOURCE_FILENAME = "summary.json"
OUTPUT_FILENAME = "summary_mean.json"


def sample_stddev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate experiment summary.json artifacts by experiment slug."
    )
    parser.add_argument("experiment_name")
    parser.add_argument("input_root", nargs="?", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--aggregate-dir-name", default=DEFAULT_AGGREGATE_DIR_NAME)
    return parser.parse_args()


def discover_summary_paths(
    input_root: Path,
    experiment_name: str,
    aggregate_dir_name: str,
) -> list[Path]:
    excluded_dir = input_root / aggregate_dir_name
    paths = [
        path
        for path in sorted(input_root.rglob(SOURCE_FILENAME))
        if excluded_dir not in path.parents
        and path.parent.name == experiment_name
        and path.parent.parent.name == "experiments"
    ]
    if not paths:
        raise FileNotFoundError(
            f"no experiments/{experiment_name}/{SOURCE_FILENAME} found under {input_root}"
        )
    return paths


def _shared_identity(payload: dict[str, Any]) -> tuple[str, str, int, int, int, float, float]:
    return (
        str(payload["task"]),
        str(payload["model"]),
        int(payload["repeats"]),
        int(payload["selected_count"]),
        int(payload["prefix_len"]),
        float(payload["g"]),
        float(payload["rho"]),
    )


def _section_keys(payload: dict[str, Any], section: str) -> list[str]:
    return sorted(str(key) for key in payload["summary"][section])


def validate_payloads(payloads: list[dict[str, Any]]) -> None:
    if not payloads:
        raise ValueError("need at least one experiment summary payload")

    expected_identity = _shared_identity(payloads[0])
    expected_metrics = _section_keys(payloads[0], "metrics")
    expected_cost = _section_keys(payloads[0], "cost")
    expected_delta = _section_keys(payloads[0], "delta")

    for payload in payloads[1:]:
        if _shared_identity(payload) != expected_identity:
            raise ValueError(
                "all experiment summaries must share task/model/repeats/selected_count/prefix_len/g/rho"
            )
        if _section_keys(payload, "metrics") != expected_metrics:
            raise ValueError("all experiment summaries must share the same metric keys")
        if _section_keys(payload, "cost") != expected_cost:
            raise ValueError("all experiment summaries must share the same cost keys")
        if _section_keys(payload, "delta") != expected_delta:
            raise ValueError("all experiment summaries must share the same delta keys")


def _mean_by_key(payloads: list[dict[str, Any]], section: str) -> dict[str, float]:
    keys = _section_keys(payloads[0], section)
    return {
        key: fmean(float(payload["summary"][section][key]) for payload in payloads)
        for key in keys
    }


def _stddev_by_key(
    payloads: list[dict[str, Any]],
    section: str,
    *,
    excluded_keys: set[str] | None = None,
) -> dict[str, float]:
    keys = [
        key
        for key in _section_keys(payloads[0], section)
        if excluded_keys is None or key not in excluded_keys
    ]
    return {
        key: sample_stddev([float(payload["summary"][section][key]) for payload in payloads])
        for key in keys
    }


def _iter_source_summaries(payloads: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for payload in payloads:
        summary = payload["summary"]
        yield {
            "run_dir": str(payload["run_dir"]),
            "output_dir": str(payload["output_dir"]),
            "metrics": {
                key: float(value) for key, value in summary["metrics"].items()
            },
            "cost": {key: float(value) for key, value in summary["cost"].items()},
            "delta": {key: float(value) for key, value in summary["delta"].items()},
        }


def build_output(
    *,
    experiment_name: str,
    input_root: Path,
    aggregate_dir: Path,
    sources: list[Path],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_payloads(payloads)
    first = payloads[0]

    return {
        "experiment": experiment_name,
        "task": str(first["task"]),
        "model": str(first["model"]),
        "repeats": int(first["repeats"]),
        "selected_count": int(first["selected_count"]),
        "prefix_len": int(first["prefix_len"]),
        "g": float(first["g"]),
        "rho": float(first["rho"]),
        "source_count": len(sources),
        "metrics_mean": _mean_by_key(payloads, "metrics"),
        "metrics_stddev": _stddev_by_key(payloads, "metrics", excluded_keys={"num_docs"}),
        "cost_mean": _mean_by_key(payloads, "cost"),
        "cost_stddev": _stddev_by_key(payloads, "cost"),
        "delta_mean": _mean_by_key(payloads, "delta"),
        "delta_stddev": _stddev_by_key(payloads, "delta"),
        "aggregation": {
            "input_root": str(input_root),
            "aggregate_dir": str(aggregate_dir),
            "source_paths": [str(path) for path in sources],
            "source_summaries": list(_iter_source_summaries(payloads)),
        },
        "date": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    aggregate_dir = input_root / args.aggregate_dir_name / args.experiment_name
    source_paths = discover_summary_paths(
        input_root=input_root,
        experiment_name=args.experiment_name,
        aggregate_dir_name=args.aggregate_dir_name,
    )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in source_paths]
    output = build_output(
        experiment_name=args.experiment_name,
        input_root=input_root,
        aggregate_dir=aggregate_dir,
        sources=source_paths,
        payloads=payloads,
    )

    aggregate_dir.mkdir(parents=True, exist_ok=True)
    output_path = aggregate_dir / OUTPUT_FILENAME
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{len(source_paths)} file(s) -> {output_path}")


if __name__ == "__main__":
    main()
