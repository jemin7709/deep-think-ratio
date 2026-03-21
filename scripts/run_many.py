from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from scripts.common import build_run_dir, build_vllm_command, load_model_settings, load_task_settings
from scripts.eval import run_evaluation
from tasks.aime24.metrics import write_postprocess_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve, evaluate, and postprocess one task across many models and seeds."
    )
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, action="append", type=Path)
    parser.add_argument("--seed", required=True, action="append", type=int)
    parser.add_argument("--limit", type=float)
    return parser.parse_args()


def start_server(model_config_path: Path, *, seed: int) -> tuple[subprocess.Popen, str]:
    model_settings = load_model_settings(model_config_path)
    command = build_vllm_command(model_settings, seed=seed)
    process = subprocess.Popen(command)
    return process, f"{model_settings.server.root_url}{model_settings.server.health_path}"


def wait_for_server(
    process: subprocess.Popen,
    health_url: str,
    *,
    timeout_seconds: float = 300.0,
    poll_interval: float = 1.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            with urlopen(health_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except URLError:
            time.sleep(poll_interval)
    raise TimeoutError(f"server did not become healthy at {health_url}")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_one(
    *,
    task_config_path: Path,
    model_config_path: Path,
    seed: int,
    limit: float | None,
) -> Path:
    task_settings = load_task_settings(task_config_path)
    model_settings = load_model_settings(model_config_path)
    run_dir = build_run_dir(task_settings.name, model_settings.name, seed)
    process, health_url = start_server(model_config_path, seed=seed)
    try:
        wait_for_server(process, health_url)
        run_evaluation(
            task_config_path=task_config_path,
            model_config_path=model_config_path,
            seed=seed,
            run_dir=run_dir,
            limit=limit,
        )
        write_postprocess_artifacts(run_dir=run_dir)
        return run_dir
    finally:
        stop_server(process)


def main() -> None:
    args = parse_args()
    for model_config_path in args.model_config:
        for seed in args.seed:
            run_one(
                task_config_path=args.task_config,
                model_config_path=model_config_path,
                seed=seed,
                limit=args.limit,
            )


if __name__ == "__main__":
    main()
