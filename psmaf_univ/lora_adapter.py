"""Small, explicit LoRA implementation for the transformer part of UNIV.

This deliberately does not depend on PEFT.  Consequently adapters are ordinary
registered modules and are always present in ``state_dict()``.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA rank/alpha must be positive and dropout must be in [0, 1)")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha) / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.lora_B(self.lora_A(self.dropout(value))) * self.scaling


def _is_transformer_linear(name: str, module: nn.Module) -> bool:
    """Match only ConvViT's token-transformer stage, never convolutional stages."""
    leaf = name.rsplit(".", 1)[-1]
    return isinstance(module, nn.Linear) and name.startswith("blocks3.") and (
        leaf in {"qkv", "proj", "fc1", "fc2"}
    )


def attach_lora(
    model: nn.Module, target_modules: list[str] | None = None, *, rank: int = 8,
    alpha: float = 16, dropout: float = 0.1,
) -> list[str]:
    """Freeze ``model`` and replace valid ConvViT transformer linears with LoRA.

    ``target_modules`` is retained for compatibility, but names are still
    validated against the transformer-only allow-list.
    """
    model.requires_grad_(False)
    requested = set(target_modules) if target_modules is not None else None
    selected = [(name, module) for name, module in model.named_modules()
                if _is_transformer_linear(name, module)
                and (requested is None or name in requested or name.rsplit(".", 1)[-1] in requested)]
    if not selected:
        raise ValueError("no valid UNIV blocks3 attention/MLP linear modules found for LoRA")
    names = []
    for name, module in selected:
        parent_name, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, LoRALinear(module, rank, alpha, dropout))
        names.append(name)
    return names


def is_lora_parameter(name: str) -> bool:
    return ".lora_A." in name or ".lora_B." in name or name.startswith(("lora_A.", "lora_B."))


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return an explicit adapter state for auditing and defensive loading."""
    return {name: value.detach().clone() for name, value in model.state_dict().items()
            if ".lora_A." in name or ".lora_B." in name}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    missing = set(state) - set(current)
    if missing:
        raise RuntimeError(f"LoRA checkpoint keys do not exist in evaluator model: {sorted(missing)}")
    current.update(state)
    model.load_state_dict(current, strict=True)
