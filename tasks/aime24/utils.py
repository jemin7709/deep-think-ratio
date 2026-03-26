from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypeAlias, cast

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics_sample import AvgAtN, PassAtK
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.utils.utils import remove_reasoning_tags
from sympy import Basic, MatrixBase


MATH_EXTRACTION_PRECISION = 6
MATH_TIMEOUT_SECONDS = 5
TASK_GOLD = (ExprExtractionConfig(), LatexExtractionConfig())
TASK_PRED = (ExprExtractionConfig(), LatexExtractionConfig())
ReasoningTags = list[tuple[str, str]]
MathVoteTarget: TypeAlias = Basic | MatrixBase | str
MathVoteTargetList: TypeAlias = list[MathVoteTarget]
RUNTIME_REASONING_TAGS: ReasoningTags | None = None
ASSISTANT_START_PREFIX = "<|start|>assistant"


def resolve_model_identity(aggregated: dict, run_dir: Path) -> str:
    config = aggregated.get("config", {})
    metadata = config.get("metadata", {})
    model_args = config.get("model_args", {})
    metadata_model_args = metadata.get("model_args", {})

    candidates = [
        model_args.get("pretrained"),
        metadata_model_args.get("pretrained"),
        config.get("model_name"),
        metadata.get("model_name"),
        run_dir.parents[1].name if len(run_dir.parents) > 1 else None,
        config.get("model"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return "unknown"


def build_gpt_oss_reasoning_tags() -> ReasoningTags:
    return [
        (
            "<|channel|>analysis<|message|>",
            "<|end|><|start|>assistant<|channel|>final<|message|>",
        )
    ]


def normalize_reasoning_tags(raw: object) -> ReasoningTags | None:
    if raw is None:
        return None
    pairs = cast(list[tuple[str, str] | list[str]], raw)
    return [(start_tag, end_tag) for start_tag, end_tag in pairs]


def configure_runtime_reasoning_tags(raw: object) -> None:
    global RUNTIME_REASONING_TAGS
    RUNTIME_REASONING_TAGS = normalize_reasoning_tags(raw)


def resolve_reasoning_tags(aggregated: dict) -> ReasoningTags | None:
    config = aggregated.get("config", {})
    metadata = config.get("metadata", {})
    for candidate in (metadata.get("reasoning_tags"), config.get("reasoning_tags")):
        tags = normalize_reasoning_tags(candidate)
        if tags is not None:
            return tags
    return None


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    answer_key = next(key for key in doc if key.lower() == "answer")
    completion = str(results[0])
    score = score_match(
        str(doc[answer_key]), completion, reasoning_tags=RUNTIME_REASONING_TAGS
    )
    return {"exact_match": int(score)}


def strip_reasoning(text: str, reasoning_tags: ReasoningTags | None) -> str:
    if reasoning_tags is None:
        return text
    cleaned = remove_reasoning_tags(text, reasoning_tags)
    return cleaned.removeprefix(ASSISTANT_START_PREFIX)


def clean_completions(
    completions: list[str],
    reasoning_tags: ReasoningTags | None = None,
) -> list[str]:
    return [strip_reasoning(text, reasoning_tags) for text in completions]


def build_doc(target: str) -> Doc:
    return Doc(query="", choices=[str(target)], gold_index=0)


def build_model_response(completions: list[str]) -> ModelResponse:
    # Sampling metrics index into `ModelResponse`, and the current LightEval
    # model output object does not preserve `text_post_processed` in `__getitem__`.
    # We therefore store already-cleaned completions directly in `text`.
    return ModelResponse(text=list(completions))


@lru_cache(maxsize=1)
def get_generic_math_pred_extraction_regexes():
    return get_extraction_regexes(
        Doc(query="", choices=[""], gold_index=0),
        TASK_PRED,
        Language.ENGLISH,
    )


@lru_cache(maxsize=1)
def make_math_matcher() -> MultilingualExtractiveMatchMetric:
    return MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        gold_extraction_target=list(TASK_GOLD),
        pred_extraction_target=list(TASK_PRED),
        precision=MATH_EXTRACTION_PRECISION,
        fallback_mode="first_match",
        extraction_mode="any_match",
        timeout_seconds=MATH_TIMEOUT_SECONDS,
    )


def extract_first_canonical_math_answer(text: str) -> str:
    extracted = extract_math_vote_targets(text)
    return str(extracted[0]) if extracted else text


def extract_math_vote_targets(text: str) -> MathVoteTargetList:
    extracted = extract_target_from_pred(
        text,
        get_generic_math_pred_extraction_regexes(),
        fallback_mode="first_match",
        extraction_mode="any_match",
        timeout_seconds=MATH_TIMEOUT_SECONDS,
    )
    return cast(MathVoteTargetList, extracted) if extracted else [text]


def math_vote_targets_match(left: MathVoteTargetList, right: MathVoteTargetList) -> bool:
    return compare_gold_target(
        left,
        right,
        precision=MATH_EXTRACTION_PRECISION,
        timeout_seconds=MATH_TIMEOUT_SECONDS,
    )


def pick_majority_math_completion(completions: list[str]) -> str:
    majority_groups: list[tuple[MathVoteTargetList, list[str]]] = []
    for completion in completions:
        extracted = extract_math_vote_targets(completion)
        for representative, grouped_completions in majority_groups:
            if math_vote_targets_match(representative, extracted):
                grouped_completions.append(completion)
                break
        else:
            majority_groups.append((extracted, [completion]))

    return max(majority_groups, key=lambda group: len(group[1]))[1][0]


def score_match(
    target: str,
    completion: str,
    *,
    reasoning_tags: ReasoningTags | None = None,
) -> float:
    cleaned = clean_completions([completion], reasoning_tags)
    return float(
        make_math_matcher().compute(build_doc(target), build_model_response(cleaned))
    )


def score_avg_at_n(
    target: str,
    completions: list[str],
    n: int,
    reasoning_tags: ReasoningTags | None = None,
) -> float:
    cleaned = clean_completions(completions[:n], reasoning_tags)
    metric = AvgAtN(n=n, sample_scoring_function=make_math_matcher())
    return float(metric.compute(build_doc(target), build_model_response(cleaned)))


def score_pass_at_k(
    target: str,
    completions: list[str],
    n: int,
    k: int,
    reasoning_tags: ReasoningTags | None = None,
) -> float:
    cleaned = clean_completions(completions[:n], reasoning_tags)
    metric = PassAtK(
        k=k,
        n=n,
        strip_strings=True,
        sample_scoring_function=make_math_matcher(),
    )
    return float(metric.compute(build_doc(target), build_model_response(cleaned)))


def extract_vote_key(
    completion: str,
    target: str,
    reasoning_tags: ReasoningTags | None = None,
) -> tuple[str, ...]:
    cleaned = clean_completions([completion], reasoning_tags)[0]
    del target
    return (extract_first_canonical_math_answer(cleaned),)


def score_maj_at_n(
    target: str,
    completions: list[str],
    n: int,
    reasoning_tags: ReasoningTags | None = None,
) -> float:
    cleaned = clean_completions(completions[:n], reasoning_tags)
    majority_completion = pick_majority_math_completion(cleaned)
    return score_match(target, majority_completion)
