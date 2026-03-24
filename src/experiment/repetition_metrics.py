"""repetition metrics utilities.

Implements seq-rep-n as defined in SJeYe0NtvH:
    1 - (unique n-grams / total n-grams).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Sequence, cast

from transformers import AutoTokenizer

from tasks.aime24.utils import clean_completions


Level = Literal["token", "word"]


def _seq_rep_n_from_symbols(symbols: Sequence[object], n: int) -> float:
    """Compute seq-rep-n from a token sequence."""
    if n <= 0:
        raise ValueError("n must be a positive integer")

    if len(symbols) < n:
        return 0.0

    total_ngrams = len(symbols) - n + 1
    if total_ngrams <= 0:
        return 0.0

    unique_ngrams = {tuple(symbols[i : i + n]) for i in range(total_ngrams)}
    return 1.0 - len(unique_ngrams) / total_ngrams


@lru_cache(maxsize=16)
def _load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


def _tokenize_for_level(
    completion: str,
    *,
    level: Level,
    model_name: str,
) -> list[str] | list[int]:
    if level == "word":
        return completion.split()

    tokenizer = _load_tokenizer(model_name)
    token_ids = tokenizer.encode(completion, add_special_tokens=False)
    return cast(list[int], list(token_ids))


def seq_rep_n_for_completion(
    completion: str,
    *,
    n: int,
    level: Level,
    model_name: str,
    reasoning_tags: list[tuple[str, str]] | None = None,
) -> float:
    """Compute seq-rep-n for a single completion."""
    cleaned = clean_completions([completion], reasoning_tags)[0]
    symbols = _tokenize_for_level(cleaned, level=level, model_name=model_name)
    return _seq_rep_n_from_symbols(symbols, n)


def mean_seq_rep_n_for_completions(
    completions: list[str],
    *,
    n: int,
    level: Level,
    model_name: str,
    reasoning_tags: list[tuple[str, str]] | None = None,
) -> float:
    """Average seq-rep-n across completions."""
    if not completions:
        return 0.0

    scores = [
        seq_rep_n_for_completion(
            completion,
            n=n,
            level=level,
            model_name=model_name,
            reasoning_tags=reasoning_tags,
        )
        for completion in completions
    ]

    return sum(scores) / len(scores)


def seq_rep_n(
    completion: str,
    *,
    n: int,
    granularity: Level,
    model_name: str,
    reasoning_tags: list[tuple[str, str]] | None = None,
) -> float:
    """단일 completion의 seq-rep-n 별칭 API."""
    return seq_rep_n_for_completion(
        completion,
        n=n,
        level=granularity,
        model_name=model_name,
        reasoning_tags=reasoning_tags,
    )


def mean_seq_rep_n(
    completions: list[str],
    *,
    n: int,
    granularity: Level,
    model_name: str,
    reasoning_tags: list[tuple[str, str]] | None = None,
) -> float:
    """completion 집합 평균 seq-rep-n 별칭 API."""
    return mean_seq_rep_n_for_completions(
        completions,
        n=n,
        level=granularity,
        model_name=model_name,
        reasoning_tags=reasoning_tags,
    )
