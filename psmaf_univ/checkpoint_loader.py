"""Conservative loading helpers for original UNIV checkpoints."""

from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Any, Iterator, overload

import torch
from torch import nn
from torch.nn import functional as F


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Select the student/model branch used by known UNIV checkpoints."""
    state, _ = _extract_state_dict_and_key(checkpoint)
    return state


def _extract_state_dict_and_key(
    checkpoint: Any, requested_key: str | None = None
) -> tuple[dict[str, torch.Tensor], str | None]:
    if isinstance(checkpoint, dict):
        if requested_key is not None:
            value = checkpoint.get(requested_key)
            if isinstance(value, dict):
                return value, requested_key
            raise ValueError(f"checkpoint contains no {requested_key!r} state dictionary")
        for key in ("student", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value, key
        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint, None
    raise ValueError("checkpoint contains no recognizable state dictionary")


@dataclass
class CheckpointLoadReport:
    """Detailed checkpoint result with PyTorch-compatible key attributes.

    Iteration yields ``missing_keys`` and ``unexpected_keys``, like PyTorch's
    ``_IncompatibleKeys`` return value, for callers which unpack that result.
    """

    missing_keys: list[str]
    unexpected_keys: list[str]
    resized_keys: list[str]
    skipped_shape_mismatch_keys: list[str]
    pos_embed_resize_info: dict[str, Any]
    model_state_key_count: int
    candidate_key_count: int
    loaded_key_count: int
    load_fraction: float
    model_parameter_count: int
    loaded_parameter_count: int
    loaded_parameter_fraction: float
    checkpoint_key: str | None

    def __iter__(self) -> Iterator[list[str]]:
        yield self.missing_keys
        yield self.unexpected_keys

    @overload
    def __getitem__(self, key: int) -> list[str]: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | str) -> Any:
        """Support both legacy tuple indexing and dictionary-like report access."""
        if isinstance(key, str):
            return getattr(self, key)
        return (self.missing_keys, self.unexpected_keys)[key]

    def __len__(self) -> int:
        return 2


def _grid_and_extra(token_count: int) -> tuple[tuple[int, int], int] | None:
    """Infer a square spatial grid and zero, one, or two special tokens."""
    candidates = []
    for extra in range(3):
        spatial_tokens = token_count - extra
        side = int(spatial_tokens**0.5)
        if spatial_tokens > 0 and side * side == spatial_tokens:
            candidates.append(((side, side), extra))
    return candidates[0] if len(candidates) == 1 else None


def resize_pos_embed_if_needed(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    key: str = "pos_embed",
) -> dict[str, Any]:
    """Resize a checkpoint positional grid in place using UNIV's bicubic scheme.

    Square source/target grids and up to two leading class/distillation tokens
    are inferred. Ambiguous or unsafe layouts are skipped rather than guessed.
    """
    info: dict[str, Any] = {
        "resized": False,
        "source_shape": None,
        "target_shape": None,
        "source_grid_size": None,
        "target_grid_size": None,
        "skipped_reason": None,
    }
    if key not in state_dict or key not in model_state_dict:
        info["skipped_reason"] = "pos_embed is not present in both checkpoint and model"
        return info

    source, target = state_dict[key], model_state_dict[key]
    info["source_shape"] = tuple(source.shape)
    info["target_shape"] = tuple(target.shape)
    if source.shape == target.shape:
        return info
    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != 1 or target.shape[0] != 1:
        info["skipped_reason"] = "pos_embed must have shape [1, N, C]"
        state_dict.pop(key)
        return info
    if source.shape[2] != target.shape[2]:
        info["skipped_reason"] = "embedding dimensions differ"
        state_dict.pop(key)
        return info

    source_layout = _grid_and_extra(source.shape[1])
    target_layout = _grid_and_extra(target.shape[1])
    if source_layout is None or target_layout is None:
        info["skipped_reason"] = "could not safely infer square spatial grids and special tokens"
        state_dict.pop(key)
        return info
    source_grid, source_extra = source_layout
    target_grid, target_extra = target_layout
    info["source_grid_size"] = source_grid
    info["target_grid_size"] = target_grid
    # Mirror UNIV's interpolation: retain leading special tokens and bicubically
    # resize only the spatial portion. When a checkpoint has fewer special tokens,
    # retain the model initialization for the absent entries instead of inventing
    # positional values. Float conversion supports CPU half tensors.
    copied_extra = min(source_extra, target_extra)
    extra_tokens = source[:, :copied_extra]
    if copied_extra < target_extra:
        initialized_extra = target[:, copied_extra:target_extra].to(dtype=source.dtype, device=source.device)
        extra_tokens = torch.cat((extra_tokens, initialized_extra), dim=1)
    spatial = source[:, source_extra:].reshape(1, *source_grid, source.shape[2]).permute(0, 3, 1, 2)
    resized = F.interpolate(spatial.float(), size=target_grid, mode="bicubic", align_corners=False)
    resized = resized.to(dtype=source.dtype, device=source.device).permute(0, 2, 3, 1).flatten(1, 2)
    state_dict[key] = torch.cat((extra_tokens, resized), dim=1)
    info["resized"] = True
    return info


def load_univ_checkpoint(
    model: nn.Module, path: str | Path, *, strict: bool = False, checkpoint_key: str | None = None
) -> CheckpointLoadReport:
    """Load a UNIV checkpoint on CPU and return a detailed loading report."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    extracted, selected_checkpoint_key = _extract_state_dict_and_key(checkpoint, checkpoint_key)
    state = {key.removeprefix("module."): value for key, value in extracted.items()}
    candidate_key_count = len(state)
    model_state = model.state_dict()
    pos_info = resize_pos_embed_if_needed(state, model_state)

    resized_keys = ["pos_embed"] if pos_info["resized"] else []
    skipped = []
    if pos_info["skipped_reason"] and pos_info["source_shape"] != pos_info["target_shape"]:
        skipped.append("pos_embed")

    for key in list(state):
        if key in model_state and state[key].shape != model_state[key].shape:
            state.pop(key)
            skipped.append(key)
    skipped = list(dict.fromkeys(skipped))
    if skipped:
        warnings.warn(f"skipped checkpoint keys with incompatible shapes: {', '.join(skipped)}", stacklevel=2)

    incompatible = model.load_state_dict(state, strict=strict)
    loaded_keys = [key for key in state if key in model_state]
    loaded_count = len(loaded_keys)
    parameter_keys = set(dict(model.named_parameters()))
    model_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    loaded_parameter_count = sum(
        model_state[key].numel() for key in loaded_keys if key in parameter_keys
    )
    return CheckpointLoadReport(
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        resized_keys=resized_keys,
        skipped_shape_mismatch_keys=skipped,
        pos_embed_resize_info=pos_info,
        model_state_key_count=len(model_state),
        candidate_key_count=candidate_key_count,
        loaded_key_count=loaded_count,
        load_fraction=loaded_count / len(model_state) if model_state else 0.0,
        model_parameter_count=model_parameter_count,
        loaded_parameter_count=loaded_parameter_count,
        loaded_parameter_fraction=(
            loaded_parameter_count / model_parameter_count if model_parameter_count else 0.0
        ),
        checkpoint_key=selected_checkpoint_key,
    )
