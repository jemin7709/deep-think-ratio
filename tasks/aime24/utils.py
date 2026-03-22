from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics_sample import AvgAtN, MajAtN, PassAtK
from lighteval.metrics.utils.extractive_match_utils import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language


MATH_EXTRACTION_PRECISION = 6
TASK_GOLD = (ExprExtractionConfig(), LatexExtractionConfig())
TASK_PRED = (ExprExtractionConfig(), LatexExtractionConfig())
GPT_OSS_FINAL_MARKER = "<|channel|>final"
GPT_OSS_MESSAGE_MARKER = "<|message|>"
GPT_OSS_END_MARKER = "<|end|>"
GPT_OSS_REASONING_BLOCK_RE = re.compile(
    r"<\|channel\|>(?:analysis(?: to=[^<]+)?|commentary to=[^<]+)<\|message\|>.*?<\|end\|>",
    re.DOTALL,
)


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
        timeout_seconds=5,
    )


def extract_first_canonical_math_answer(text: str) -> str:
    extracted = extract_target_from_pred(
        text,
        get_generic_math_pred_extraction_regexes(),
        fallback_mode="first_match",
        extraction_mode="any_match",
        timeout_seconds=5,
    )
    return str(extracted[0]) if extracted else text


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
    del target
    return (extract_first_canonical_math_answer(cleaned),)


def score_maj_at_n(target: str, completions: list[str], n: int, profile: str) -> float:
    cleaned = clean_completions(completions[:n], profile)
    metric = MajAtN(n=n)
    metric.normalize = extract_first_canonical_math_answer
    return float(metric.compute(build_doc(target), build_model_response(cleaned)))


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
