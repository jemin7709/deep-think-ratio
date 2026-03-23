from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from lm_eval.config.evaluate_config import EvaluatorConfig
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode


TASKS_DIR = Path("tasks")
RESULTS_DIR = Path("results")
DEFAULT_MODEL_BACKEND = "vllm"
ALLOWED_MODEL_KEYS = {
    "model",
    "model_args",
    "batch_size",
    "max_batch_size",
    "device",
    "reasoning_tags",
    "apply_chat_template",
    "fewshot_as_multiturn",
    "gen_kwargs",
    "predict_only",
}


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
class ModelSettings:
    config_path: Path
    name: str
    model: str
    model_args: dict[str, Any] = field(default_factory=dict)
    batch_size: int | str | None = None
    max_batch_size: int | None = None
    device: str | None = None
    reasoning_tags: list[list[str]] | None = None
    apply_chat_template: bool | str = True
    fewshot_as_multiturn: bool | None = True
    gen_kwargs: dict[str, Any] = field(default_factory=dict)
    predict_only: bool = True


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
    raw = load_yaml(model_config_path, loader=TaggedYamlLoader)
    if set(raw) == {"server", "harness"}:
        raise ValueError(
            f"{model_config_path} uses the removed server/harness schema; "
            "migrate it to the vllm-only model schema"
        )

    unknown_keys = set(raw) - ALLOWED_MODEL_KEYS
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"{model_config_path} has unsupported keys: {unknown}")

    backend = raw.get("model")
    if backend != DEFAULT_MODEL_BACKEND:
        raise ValueError(
            f"{model_config_path} must declare model: {DEFAULT_MODEL_BACKEND}"
        )

    model_args_raw = raw.get("model_args", {})
    if not isinstance(model_args_raw, dict):
        raise ValueError("model_args must be a mapping")
    model_args = dict(model_args_raw)
    if not isinstance(model_args.get("pretrained"), str) or not model_args["pretrained"]:
        raise ValueError("model_args.pretrained must be a non-empty string")
    chat_template_args = model_args.get("chat_template_args")
    if chat_template_args is not None and not isinstance(chat_template_args, dict):
        raise ValueError("model_args.chat_template_args must be a mapping")
    reasoning_tags = raw.get("reasoning_tags")
    if reasoning_tags is not None:
        reasoning_tags = [list(item) for item in reasoning_tags]

    batch_size = raw.get("batch_size")
    if batch_size is not None and batch_size != "auto":
        batch_size = int(batch_size)
    max_batch_size = raw.get("max_batch_size")
    if max_batch_size is not None:
        max_batch_size = int(max_batch_size)
    apply_chat_template = raw.get("apply_chat_template", True)
    fewshot_as_multiturn = raw.get("fewshot_as_multiturn", True)
    gen_kwargs_raw = raw.get("gen_kwargs", {})
    if not isinstance(gen_kwargs_raw, dict):
        raise ValueError("gen_kwargs must be a mapping")
    gen_kwargs = dict(gen_kwargs_raw)
    predict_only = bool(raw.get("predict_only", True))

    return ModelSettings(
        config_path=model_config_path,
        name=model_config_path.stem,
        model=backend,
        model_args=model_args,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=raw.get("device"),
        reasoning_tags=reasoning_tags,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        gen_kwargs=gen_kwargs,
        predict_only=predict_only,
    )


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
    *,
    seed: int,
) -> dict[str, Any]:
    resolved_model_args = deepcopy(model_settings.model_args)
    resolved_model_args["seed"] = seed
    return {
        "task_name": task_settings.name,
        "task_config": str(task_settings.config_path),
        "model_name": model_settings.name,
        "model_config": str(model_settings.config_path),
        "repeats": task_settings.repeats,
        "seed": seed,
        "model": model_settings.model,
        "model_args": resolved_model_args,
        "batch_size": model_settings.batch_size,
        "max_batch_size": model_settings.max_batch_size,
        "reasoning_tags": deepcopy(model_settings.reasoning_tags),
        "apply_chat_template": model_settings.apply_chat_template,
        "fewshot_as_multiturn": model_settings.fewshot_as_multiturn,
        "gen_kwargs": deepcopy(model_settings.gen_kwargs),
        "predict_only": model_settings.predict_only,
    }


def build_evaluator_config(
    task_settings: TaskSettings,
    model_settings: ModelSettings,
    *,
    run_dir: Path,
    seed: int,
    limit: float | None,
) -> EvaluatorConfig:
    model_args = deepcopy(model_settings.model_args)
    model_args["seed"] = seed
    config_kwargs: dict[str, Any] = dict(
        model=model_settings.model,
        model_args=model_args,
        tasks=[task_settings.name],
        num_fewshot=task_settings.num_fewshot,
        max_batch_size=model_settings.max_batch_size,
        device=model_settings.device,
        limit=limit,
        log_samples=True,
        output_path=str(build_tracker_output_path(run_dir)),
        predict_only=model_settings.predict_only,
        include_path=str(task_settings.include_path),
        apply_chat_template=model_settings.apply_chat_template,
        fewshot_as_multiturn=model_settings.fewshot_as_multiturn,
        gen_kwargs=deepcopy(model_settings.gen_kwargs),
        seed=[seed, seed, seed, seed],
        metadata=build_metadata(
            task_settings,
            model_settings,
            seed=seed,
        ),
    )
    config_kwargs["batch_size"] = model_settings.batch_size
    config = EvaluatorConfig(**config_kwargs)
    return config._parse_dict_args()._configure()


def find_task_config_path(task_name: str, *, tasks_root: Path = TASKS_DIR) -> Path:
    for path in tasks_root.rglob("*.yaml"):
        raw = load_yaml(path, loader=TaggedYamlLoader)
        if raw.get("task") == task_name:
            return path
    raise FileNotFoundError(f"could not find a task config for {task_name}")
