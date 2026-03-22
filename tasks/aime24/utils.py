from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from sympy import Basic, MatrixBase
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics_sample import AvgAtN, PassAtK
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    ExtractionTarget,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.metrics.utils.math_comparison import compare_gold_target
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


MATH_EXTRACTION_PRECISION = 6
TASK_GOLD = (ExprExtractionConfig(), LatexExtractionConfig())
TASK_PRED = (ExprExtractionConfig(), LatexExtractionConfig())
NO_EXTRACTION_KEY = ("__NO_EXTRACTION__",)
GPT_OSS_FINAL_MARKER = "<|channel|>final"
GPT_OSS_MESSAGE_MARKER = "<|message|>"
GPT_OSS_END_MARKER = "<|end|>"
GPT_OSS_REASONING_BLOCK_RE = re.compile(
    r"<\|channel\|>(?:analysis(?: to=[^<]+)?|commentary to=[^<]+)<\|message\|>.*?<\|end\|>",
    re.DOTALL,
)
ExtractionValue = Basic | MatrixBase | str


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


def resolve_reasoning_profile(model_identity: str) -> str:
    normalized = model_identity.lower()
    if "gpt-oss" in normalized:
        return "gptoss_harmony"
    if "deepseek" in normalized:
        return "deepseek_think_end"
    if "qwen" in normalized:
        return "qwen_think_end"
    return "identity"


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    answer_key = next(key for key in doc if key.lower() == "answer")
    completion = str(results[0])
    profile = infer_profile_from_completion(completion)
    score = score_match(str(doc[answer_key]), completion, profile=profile)
    return {"exact_match": int(score)}


def infer_profile_from_completion(completion: str) -> str:
    if GPT_OSS_FINAL_MARKER in completion or "<|channel|>analysis" in completion:
        return "gptoss_harmony"
    if "</think>" in completion or "<think>" in completion:
        return "deepseek_think_end"
    return "identity"


def strip_reasoning_with_profile(text: str, profile: str) -> str | None:
    if profile == "gptoss_harmony":
        return strip_gpt_oss_reasoning(text)
    if profile in {"deepseek_think_end", "qwen_think_end"}:
        return strip_think_end_reasoning(text)
    return text


def clean_completions(completions: list[str], profile: str) -> list[str]:
    return [
        cleaned if cleaned is not None else ""
        for cleaned in (strip_reasoning_with_profile(text, profile) for text in completions)
    ]


def build_doc(target: str) -> Doc:
    return Doc(query="", choices=[str(target)], gold_index=0)


def build_model_response(completions: list[str]) -> ModelResponse:
    # Sampling metrics index into `ModelResponse`, and the current LightEval
    # model output object does not preserve `text_post_processed` in `__getitem__`.
    # We therefore store already-cleaned completions directly in `text`.
    return ModelResponse(text=list(completions))


@lru_cache(maxsize=1)
def make_math_matcher() -> MultilingualExtractiveMatchMetric:
    return MultilingualExtractiveMatchMetric(
        language=Language.ENGLISH,
        gold_extraction_target=list(TASK_GOLD),
        pred_extraction_target=list(TASK_PRED),
        precision=MATH_EXTRACTION_PRECISION,
        fallback_mode="first_match",
        extraction_mode="any_match",
        timeout_seconds=5,
    )


@lru_cache(maxsize=None)
def get_math_extraction_regexes(
    target: str,
) -> tuple[
    list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
    list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
]:
    doc = build_doc(target)
    return (
        get_extraction_regexes(doc, TASK_GOLD, Language.ENGLISH),
        get_extraction_regexes(doc, TASK_PRED, Language.ENGLISH),
    )


def extract_gold_targets(target: str) -> list[ExtractionValue]:
    gold_regexes, _ = get_math_extraction_regexes(str(target))
    return list(
        extract_target_from_pred(
            str(target),
            gold_regexes,
            fallback_mode="first_match",
            extraction_mode="any_match",
            timeout_seconds=5,
        )
    )


def extract_prediction_targets(target: str, completion: str) -> list[ExtractionValue]:
    _, pred_regexes = get_math_extraction_regexes(str(target))
    return list(
        extract_target_from_pred(
            completion,
            pred_regexes,
            fallback_mode="first_match",
            extraction_mode="any_match",
            timeout_seconds=5,
        )
    )


def score_match(target: str, completion: str, *, profile: str) -> float:
    cleaned = clean_completions([completion], profile)
    return float(make_math_matcher().compute(build_doc(target), build_model_response(cleaned)))


def score_avg_at_n(target: str, completions: list[str], n: int, profile: str) -> float:
    cleaned = clean_completions(completions[:n], profile)
    metric = AvgAtN(n=n, sample_scoring_function=make_math_matcher())
    return float(metric.compute(build_doc(target), build_model_response(cleaned)))


def score_pass_at_k(target: str, completions: list[str], n: int, k: int, profile: str) -> float:
    cleaned = clean_completions(completions[:n], profile)
    metric = PassAtK(
        k=k,
        n=n,
        strip_strings=True,
        sample_scoring_function=make_math_matcher(),
    )
    return float(metric.compute(build_doc(target), build_model_response(cleaned)))


def extract_vote_key(completion: str, target: str, profile: str) -> tuple[str, ...]:
    cleaned = clean_completions([completion], profile)[0]
    extracted = extract_prediction_targets(target, cleaned)
    if not extracted:
        return NO_EXTRACTION_KEY
    return tuple(str(item) for item in extracted)


def score_maj_at_n(target: str, completions: list[str], n: int, profile: str) -> float:
    cleaned = clean_completions(completions[:n], profile)
    gold_targets = extract_gold_targets(target)
    vote_counts: dict[tuple[str, ...], int] = {}
    first_seen: dict[tuple[str, ...], int] = {}
    extracted_by_key: dict[tuple[str, ...], list[ExtractionValue]] = {}

    for index, completion in enumerate(cleaned):
        extracted = extract_prediction_targets(target, completion)
        key = tuple(str(item) for item in extracted) if extracted else NO_EXTRACTION_KEY
        if key not in vote_counts:
            vote_counts[key] = 0
            first_seen[key] = index
            extracted_by_key[key] = extracted
        vote_counts[key] += 1

    winner = max(vote_counts, key=lambda key: (vote_counts[key], -first_seen[key]))
    return float(
        compare_gold_target(
            gold_targets,
            extracted_by_key[winner],
            precision=MATH_EXTRACTION_PRECISION,
            timeout_seconds=5,
        )
    )


def strip_gpt_oss_reasoning(text: str) -> str | None:
    final_index = text.rfind(GPT_OSS_FINAL_MARKER)
    if final_index >= 0:
        message_index = text.find(GPT_OSS_MESSAGE_MARKER, final_index)
        if message_index >= 0:
            content = text[message_index + len(GPT_OSS_MESSAGE_MARKER) :]
            return content.split(GPT_OSS_END_MARKER, 1)[0]

    if "<|channel|>" not in text and "<|start|>" not in text and GPT_OSS_END_MARKER not in text:
        return text

    cleaned = GPT_OSS_REASONING_BLOCK_RE.sub("", text)
    if cleaned != text:
        return cleaned.split(GPT_OSS_END_MARKER, 1)[0]
    return text


def strip_think_end_reasoning(text: str) -> str | None:
    _, start_token, remainder = text.partition("<think>")
    candidate = remainder if start_token else text
    if "</think>" not in candidate:
        return None
    _, _, content = candidate.partition("</think>")
    return content or None
