"""`deep-think-tokens`의 핵심 hook 코드를 거의 그대로 가져온다."""

from __future__ import annotations

import abc
import re
from collections import OrderedDict, defaultdict
from functools import wraps
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


class Tracker(abc.ABC):
    """Base class for tracking model behavior."""

    def __init__(
        self,
        model: Any,
        clear_on_generate: bool = True,
    ) -> None:
        self.model = model
        self.clear_on_generate = clear_on_generate
        self.original_generate = self.model.generate
        if clear_on_generate:
            self._patch_generate()
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self.aggregate_hook = model.register_forward_hook(self._get_aggregate_hook())

    def _patch_generate(self) -> None:
        @wraps(self.model.generate)
        def new_generate(*args, **kwargs):
            self.clear()
            return self.original_generate(*args, **kwargs)

        self.model.generate = new_generate

    @abc.abstractmethod
    def _get_aggregate_hook(self) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def collect(self):
        raise NotImplementedError

    @abc.abstractmethod
    def clear(self) -> None:
        raise NotImplementedError

    def detach(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.aggregate_hook.remove()
        self.model.generate = self.original_generate


class DeepThinkingTokensTracker(Tracker):
    """Tracks deep thinking tokens for each layer, for each token."""

    def __init__(
        self,
        model: Any,
        logits_hooks: OrderedDict[str, tuple[Any, torch.utils.hooks.RemovableHandle]],
        last_layer_name: str,
        clear_on_generate: bool = True,
    ) -> None:
        super().__init__(model, clear_on_generate)
        self.logits_hooks = logits_hooks
        self.last_layer_name = last_layer_name
        for _, (_, handle) in logits_hooks.items():
            self.hooks.append(handle)
        self.divergences: dict[str, list[torch.Tensor]] = defaultdict(list)

    def _get_aggregate_hook(self):
        @torch.no_grad()
        def aggregate_hook(*_):
            final_layer, _ = self.logits_hooks[self.last_layer_name]
            final_logits = torch.as_tensor(final_layer._logits)
            final_log_probs = torch.log_softmax(final_logits, dim=-1)
            final_probs = torch.softmax(final_logits, dim=-1)

            for name, (layer, _) in self.logits_hooks.items():
                if name == self.last_layer_name:
                    continue
                layer_logits = torch.as_tensor(layer._logits)
                log_probs = torch.log_softmax(layer_logits, dim=-1)
                probs = torch.softmax(layer_logits, dim=-1)
                mixture = torch.logsumexp(
                    torch.stack([final_log_probs, log_probs], dim=0),
                    dim=0,
                ) - torch.log(torch.tensor(2.0, device=probs.device))
                divergence = (
                    (
                        F.kl_div(mixture, probs, reduction="none")
                        + F.kl_div(mixture, final_probs, reduction="none")
                    )
                    / 2
                ).sum(dim=-1)
                self.divergences[name].append(divergence.detach().cpu())

            for layer, _ in self.logits_hooks.values():
                if hasattr(layer, "_logits"):
                    delattr(layer, "_logits")

        return aggregate_hook

    def collect(self) -> dict[str, torch.Tensor]:
        return {
            name: torch.cat(divergences, dim=1)
            for name, divergences in self.divergences.items()
            if divergences
        }

    def clear(self) -> None:
        self.divergences.clear()


def add_deep_thinking_tokens_hooks(
    model: PreTrainedModel,
    clear_on_generate: bool = True,
    module_names: list[str] | None = None,
) -> DeepThinkingTokensTracker:
    """Registers forward hooks on transformer layers."""

    @torch.no_grad()
    def layer_hook(module: nn.Module, _: tuple[Any, ...], output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        module._logits = model.lm_head(hidden).detach()

    hooks: OrderedDict[
        str, tuple[nn.Module, torch.utils.hooks.RemovableHandle]
    ] = OrderedDict()
    last_layer_name = module_names[-1] if module_names else ""

    for name, layer in model.named_modules():
        if module_names and name in module_names:
            hooks[name] = (layer, layer.register_forward_hook(layer_hook))
        elif re.match(r".*\.layers\.\d+$", name):
            hooks[name] = (layer, layer.register_forward_hook(layer_hook))
            last_layer_name = name

    if not hooks or not last_layer_name:
        raise ValueError("failed to locate transformer layers for deep-think tracking")

    return DeepThinkingTokensTracker(
        model,
        hooks,
        last_layer_name=last_layer_name,
        clear_on_generate=clear_on_generate,
    )
