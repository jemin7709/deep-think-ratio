"""`postprocess_*.json` 산출물을 여러 run에 걸쳐 평균낸다.

원본 저장소의 `average_results.py`는 `results.json` 내부 repeat metric을 직접 평균냈다.
현재 프로젝트는 run마다 `postprocess_*.json`을 남기므로, 이 모듈은 그 형식에 맞춰
평균 metric과 소스 메타데이터를 함께 저장한다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


SOURCE_GLOB = "postprocess_*.json"
OUTPUT_FILENAME = "postprocess_mean.json"


def sample_stddev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average postprocess_*.json artifacts across repeated runs."
    )
    parser.add_argument("input_path", type=Path)
    return parser.parse_args()


def find_sources(input_path: Path) -> tuple[list[Path], Path]:
    input_path = input_path.resolve()

    if input_path.is_file():
        if input_path.name == OUTPUT_FILENAME:
            raise ValueError(f"{OUTPUT_FILENAME} is an aggregation output, not a source input")
        return [input_path], input_path.parent

    sources = sorted(
        path for path in input_path.rglob(SOURCE_GLOB) if path.name != OUTPUT_FILENAME
    )
    if not sources:
        raise FileNotFoundError(f"no {SOURCE_GLOB} found under {input_path}")
    return sources, input_path


def _shared_identity(payload: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(payload["task"]),
        str(payload["model"]),
        int(payload["repeats"]),
        int(payload["k"]),
    )


def _metric_keys(payload: dict[str, Any], section: str) -> list[str]:
    return sorted(str(key) for key in payload.get(section, {}))


def validate_payloads(payloads: list[dict[str, Any]]) -> None:
    if not payloads:
        raise ValueError("need at least one postprocess payload")

    expected_identity = _shared_identity(payloads[0])
    expected_metrics = _metric_keys(payloads[0], "metrics")

    for payload in payloads[1:]:
        if _shared_identity(payload) != expected_identity:
            raise ValueError("all postprocess payloads must share task/model/repeats/k")
        if _metric_keys(payload, "metrics") != expected_metrics:
            raise ValueError("all postprocess payloads must share the same metric keys")


def _mean_by_key(
    payloads: list[dict[str, Any]],
    section: str,
) -> dict[str, float]:
    keys = _metric_keys(payloads[0], section)
    return {
        key: fmean(float(payload[section][key]) for payload in payloads)
        for key in keys
    }


def _stddev_by_metric(payloads: list[dict[str, Any]]) -> dict[str, float]:
    keys = [key for key in _metric_keys(payloads[0], "metrics") if key != "num_docs"]
    return {
        key: sample_stddev([float(payload["metrics"][key]) for payload in payloads])
        for key in keys
    }


def _iter_source_metrics(payloads: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for payload in payloads:
        yield {
            "run_dir": str(payload.get("run_dir", "")),
            "metrics": {key: float(value) for key, value in payload["metrics"].items()},
        }


def build_output(
    input_path: Path,
    sources: list[Path],
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    validate_payloads(payloads)
    first = payloads[0]

    return {
        "task": str(first["task"]),
        "model": str(first["model"]),
        "repeats": int(first["repeats"]),
        "k": int(first["k"]),
        "source_count": len(sources),
        "num_docs_per_source": [int(payload["metrics"]["num_docs"]) for payload in payloads],
        "metrics_mean": _mean_by_key(payloads, "metrics"),
        "metrics_stddev": _stddev_by_metric(payloads),
        "aggregation": {
            "input_path": str(input_path),
            "mode": "single_postprocess_file" if len(sources) == 1 else "directory_tree",
            "source_paths": [str(path) for path in sources],
            "source_metrics": list(_iter_source_metrics(payloads)),
        },
        "date": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    input_path = parse_args().input_path
    sources, output_dir = find_sources(input_path)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    output = build_output(input_path.resolve(), sources, payloads)

    output_path = output_dir / OUTPUT_FILENAME
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{len(sources)} file(s) -> {output_path}")


if __name__ == "__main__":
    main()
