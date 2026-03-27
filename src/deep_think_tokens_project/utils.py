"""`deep-think-tokens`의 `utils.py`에서 필요한 부분만 단순히 가져온다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor


DEFAULT_G = 0.5
DEFAULT_P = 0.9


@dataclass(frozen=True)
class DeepThinkingResult:
    """DTR 계산 결과."""

    dtr: float
    first_deep_layer: Tensor
    deep_mask: Tensor


def validate_divergence_inputs(
    *,
    divergence_matrix: Tensor,
    g: float,
    p: float,
) -> None:
    if divergence_matrix.ndim != 2:
        raise ValueError(
            "divergence_matrix must be 2-dimensional [num_layers, num_tokens]"
        )
    num_layers, num_tokens = divergence_matrix.shape
    if num_layers < 1:
        raise ValueError("divergence_matrix must contain at least one layer row")
    if num_tokens < 1:
        raise ValueError("divergence_matrix must contain at least one token column")
    if not 0.0 <= g <= 1.0:
        raise ValueError(f"g must be in the interval [0, 1], got {g}")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in the interval (0, 1], got {p}")


def validate_prefix_len(prefix_len: int) -> None:
    if prefix_len < 1:
        raise ValueError(f"prefix_len must be >= 1, got {prefix_len}")


def div_to_matrix(divergences: dict[str, Tensor]) -> Tensor:
    """Layer별 divergence dict를 [num_layers, num_tokens] 행렬로 바꾼다."""
    if not divergences:
        raise ValueError("divergences must contain at least one tracked layer")
    rows = [value.detach().cpu() for value in divergences.values()]
    matrix = torch.cat(rows, dim=0).float()
    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        matrix = torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix


def compute_dtr_from_divergence_matrix(
    divergence_matrix: Tensor | np.ndarray | dict[Any, Tensor],
    *,
    g: float = DEFAULT_G,
    p: float = DEFAULT_P,
) -> DeepThinkingResult:
    """`deep-think-tokens`의 layer-major 정의를 그대로 따른다."""
    if isinstance(divergence_matrix, dict):
        matrix = div_to_matrix(divergences=divergence_matrix)
    elif isinstance(divergence_matrix, np.ndarray):
        matrix = torch.from_numpy(divergence_matrix).float()
    else:
        matrix = divergence_matrix.float()

    validate_divergence_inputs(divergence_matrix=matrix, g=g, p=p)
    num_layers, num_tokens = matrix.shape
    depth_threshold = int(num_layers * p)
    cummin_divergence = torch.cummin(matrix, dim=0).values
    settled_mask = cummin_divergence <= g

    layer_indices = torch.arange(num_layers, device=matrix.device)
    layer_indices = layer_indices.unsqueeze(1).expand(-1, num_tokens)
    masked = torch.where(
        settled_mask,
        layer_indices,
        torch.full_like(layer_indices, num_layers),
    )
    first_deep_layer = masked.min(dim=0).values
    deep_mask = first_deep_layer >= depth_threshold
    dtr = deep_mask.float().mean().item()
    return DeepThinkingResult(
        dtr=dtr,
        first_deep_layer=first_deep_layer,
        deep_mask=deep_mask,
    )
