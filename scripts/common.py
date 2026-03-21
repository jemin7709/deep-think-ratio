from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from lm_eval.config.evaluate_config import EvaluatorConfig
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode


TASKS_DIR = Path("tasks")
RESULTS_DIR = Path("results")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_HEALTH_PATH = "/health"
ALLOWED_BACKENDS = {"local-completions", "local-chat-completions"}


class TaggedYamlLoader(yaml.SafeLoader):
    pass


def _construct_tagged_value(
    loader: TaggedYamlLoader, tag_suffix: str, node: Node
) -> object:
    if isinstance(node, ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"unsupported YAML node: {type(node).__name__}")


TaggedYamlLoader.add_multi_constructor("!", _construct_tagged_value)


@dataclass(frozen=True)
class TaskSettings:
    config_path: Path
    include_path: Path
    name: str
    repeats: int
    num_fewshot: int | None


@dataclass(frozen=True)
class ServerConfig:
    model: str
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    seed: int | None = None
    health_path: str = DEFAULT_HEALTH_PATH
    extra_args: dict[str, Any] = field(default_factory=dict)

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class HarnessConfig:
    model: str
    model_args: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    max_batch_size: int | None = None
    device: str | None = None
    apply_chat_template: bool | str = True
    fewshot_as_multiturn: bool | None = True
    gen_kwargs: dict[str, Any] = field(default_factory=dict)
    predict_only: bool = True


@dataclass(frozen=True)
class ModelSettings:
    config_path: Path
    name: str
    server: ServerConfig
    harness: HarnessConfig


def load_yaml(path: Path, *, loader: type[yaml.SafeLoader] = yaml.SafeLoader) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing yaml file: {path}")
    data = yaml.load(path.read_text(encoding="utf-8"), Loader=loader)
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping in {path}")
    return data


def load_task_settings(task_config_path: Path) -> TaskSettings:
    raw = load_yaml(task_config_path, loader=TaggedYamlLoader)
    task_name = raw.get("task")
    repeats = raw.get("repeats")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError(f"{task_config_path} must define a non-empty task")
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"{task_config_path} must define repeats >= 1")
    num_fewshot = raw.get("num_fewshot")
    if num_fewshot is not None and not isinstance(num_fewshot, int):
        raise ValueError(f"{task_config_path} num_fewshot must be an integer")
    return TaskSettings(
        config_path=task_config_path,
        include_path=task_config_path.parent.parent,
        name=task_name,
        repeats=repeats,
        num_fewshot=num_fewshot,
    )


def load_model_settings(model_config_path: Path) -> ModelSettings:
    raw = load_yaml(model_config_path)
    top_level_keys = set(raw)
    if top_level_keys != {"server", "harness"}:
        raise ValueError(
            f"{model_config_path} must use the new server/harness schema exactly"
        )

    server_raw = raw["server"]
    harness_raw = raw["harness"]
    if not isinstance(server_raw, dict) or not isinstance(harness_raw, dict):
        raise ValueError(f"{model_config_path} server and harness must both be mappings")

    server = parse_server_config(server_raw)
    harness = parse_harness_config(server, harness_raw)
    return ModelSettings(
        config_path=model_config_path,
        name=model_config_path.stem,
        server=server,
        harness=harness,
    )


def parse_server_config(raw: dict[str, Any]) -> ServerConfig:
    if not isinstance(raw.get("model"), str) or not raw["model"]:
        raise ValueError("server.model must be a non-empty string")

    extra_args = dict(raw)
    model = str(extra_args.pop("model"))
    host = str(extra_args.pop("host", DEFAULT_HOST))
    port = int(extra_args.pop("port", DEFAULT_PORT))
    seed = extra_args.pop("seed", None)
    if seed is not None:
        seed = int(seed)
    health_path = str(extra_args.pop("health_path", DEFAULT_HEALTH_PATH))

    return ServerConfig(
        model=model,
        host=host,
        port=port,
        seed=seed,
        health_path=health_path,
        extra_args=extra_args,
    )


def parse_harness_config(server: ServerConfig, raw: dict[str, Any]) -> HarnessConfig:
    backend = str(raw.get("model", "local-completions"))
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(f"harness.model must be one of {sorted(ALLOWED_BACKENDS)}")

    model_args = dict(raw.get("model_args", {}))
    if not isinstance(model_args, dict):
        raise ValueError("harness.model_args must be a mapping")

    model_args.setdefault("model", server.model)
    model_args.setdefault("base_url", build_base_url(server, backend))

    batch_size = int(raw.get("batch_size", 1))
    max_batch_size = raw.get("max_batch_size")
    apply_chat_template = raw.get("apply_chat_template", True)
    fewshot_as_multiturn = raw.get("fewshot_as_multiturn", True)
    gen_kwargs = dict(raw.get("gen_kwargs", {}))
    if not isinstance(gen_kwargs, dict):
        raise ValueError("harness.gen_kwargs must be a mapping")
    predict_only = bool(raw.get("predict_only", True))

    if backend == "local-chat-completions":
        if apply_chat_template is not True:
            raise ValueError(
                "local-chat-completions requires harness.apply_chat_template=true"
            )
        if "chat_template" not in server.extra_args:
            raise ValueError(
                "local-chat-completions requires server.chat_template to be set"
            )
        batch_size = 1
        if max_batch_size is not None:
            max_batch_size = 1

    return HarnessConfig(
        model=backend,
        model_args=model_args,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=raw.get("device"),
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        gen_kwargs=gen_kwargs,
        predict_only=predict_only,
    )


def build_base_url(server: ServerConfig, backend: str) -> str:
    endpoint = (
        "/v1/chat/completions"
        if backend == "local-chat-completions"
        else "/v1/completions"
    )
    return f"{server.root_url}{endpoint}"


def timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_dir(
    task_name: str,
    model_name: str,
    seed: int,
    *,
    root: Path = RESULTS_DIR,
    timestamp: str | None = None,
) -> Path:
    stamp = timestamp or timestamp_now()
    run_dir = root / task_name / model_name / str(seed) / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_tracker_output_path(run_dir: Path) -> Path:
    return run_dir / "results.json"


def build_metadata(
    task_settings: TaskSettings,
    model_settings: ModelSettings,
    seed: int,
) -> dict[str, Any]:
    return {
        "task_name": task_settings.name,
        "task_config": str(task_settings.config_path),
        "model_name": model_settings.name,
        "model_config": str(model_settings.config_path),
        "repeats": task_settings.repeats,
        "framework_seed": seed,
        "server_seed": seed,
        "request_seed": seed,
    }


def build_evaluator_config(
    task_settings: TaskSettings,
    model_settings: ModelSettings,
    *,
    run_dir: Path,
    seed: int,
    limit: float | None,
) -> EvaluatorConfig:
    harness = model_settings.harness
    config = EvaluatorConfig(
        model=harness.model,
        model_args={**harness.model_args, "seed": seed},
        tasks=[task_settings.name],
        num_fewshot=task_settings.num_fewshot,
        batch_size=harness.batch_size,
        max_batch_size=harness.max_batch_size,
        device=harness.device,
        limit=limit,
        log_samples=True,
        output_path=str(build_tracker_output_path(run_dir)),
        predict_only=harness.predict_only,
        include_path=str(task_settings.include_path),
        apply_chat_template=harness.apply_chat_template,
        fewshot_as_multiturn=harness.fewshot_as_multiturn,
        gen_kwargs=harness.gen_kwargs,
        seed=[seed, seed, seed, seed],
        metadata=build_metadata(task_settings, model_settings, seed),
    )
    return config._parse_dict_args()._configure()


def build_vllm_command(model_settings: ModelSettings, *, seed: int) -> list[str]:
    server = model_settings.server
    command = [
        "vllm",
        "serve",
        server.model,
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--seed",
        str(seed),
    ]
    for key, value in server.extra_args.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if isinstance(value, (dict, list)):
            command.extend([flag, json.dumps(value, ensure_ascii=False)])
            continue
        command.extend([flag, str(value)])
    return command


def find_task_config_path(task_name: str, *, tasks_root: Path = TASKS_DIR) -> Path:
    for path in tasks_root.rglob("*.yaml"):
        raw = load_yaml(path, loader=TaggedYamlLoader)
        if raw.get("task") == task_name:
            return path
    raise FileNotFoundError(f"could not find a task config for {task_name}")
