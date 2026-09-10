"""Conservative loading helpers for original UNIV checkpoints."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Select the student/model branch used by known UNIV checkpoints."""
    if isinstance(checkpoint, dict):
        for key in ("student", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
    raise ValueError("checkpoint contains no recognizable state dictionary")


def load_univ_checkpoint(model: nn.Module, path: str | Path, *, strict: bool = False):
    """Load weights on CPU and return PyTorch's incompatibility report."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = {key.removeprefix("module."): value for key, value in extract_state_dict(checkpoint).items()}
    return model.load_state_dict(state, strict=strict)
