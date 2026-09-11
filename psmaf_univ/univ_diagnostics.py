"""Runtime diagnostics for the checked-in UNIV ConvMAE implementation.

This module deliberately stops at observation: it constructs the original model,
captures intermediate activations, and reports checkpoint compatibility without
adding a detector or changing the upstream source.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import importlib
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor, nn

from .checkpoint_loader import CheckpointLoadReport, load_univ_checkpoint
from .compat import apply_numpy_legacy_aliases


DEFAULT_PROBE_MODULES = (
    "patch_embed1",
    "blocks1.1",
    "patch_embed2",
    "blocks2.1",
    "patch_embed3",
    "patch_embed4",
    "blocks3.10",
    "norm",
)


def build_original_univ(source_root: str | Path | None = None) -> nn.Module:
    """Construct the exact ConvMAE factory used by ``pretrain_mcmae.py``.

    The upstream directory has a hyphen in its name and is intentionally not
    repackaged.  Adding it to ``sys.path`` keeps imports inside the snapshot
    working while leaving its files untouched.
    """
    root = (
        Path(source_root)
        if source_root is not None
        else Path(__file__).resolve().parents[1] / "UNIV-main"
    )
    root = root.resolve()
    if not (root / "models/backbone/mcmae/models_convmae.py").is_file():
        raise FileNotFoundError(f"UNIV model source not found under {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # The original UNIV snapshot uses aliases removed in NumPy 1.24. Apply the
    # focused compatibility shim immediately before importing that source.
    apply_numpy_legacy_aliases()
    module = importlib.import_module("models.backbone.mcmae.models_convmae")
    return module.convmae_convvit_base_patch16()


def model_inventory(model: nn.Module) -> dict[str, Any]:
    """Return architecture and parameter facts suitable for JSON output."""
    parameters = list(model.parameters())
    named_modules = dict(model.named_modules())
    named_parameters = dict(model.named_parameters())
    patch_embeddings = [
        name
        for name, module in named_modules.items()
        if name
        and (
            name.rsplit(".", 1)[-1].lower().startswith("patch_embed")
            or "patchembed" in type(module).__name__.lower()
        )
    ]
    positional_embeddings = [name for name in named_parameters if "pos_embed" in name.lower()]
    block_groups = {
        name: len(module)
        for name, module in named_modules.items()
        if name
        and "block" in name.lower()
        and "decoder" not in name.lower()
        and isinstance(module, (nn.ModuleList, nn.Sequential))
    }
    norms = [
        name for name, module in named_modules.items()
        if name and ("norm" in name.lower() or isinstance(module, nn.LayerNorm))
    ]
    attentions = [
        name for name, module in named_modules.items()
        if name and ("attn" in name.lower() or "attention" in type(module).__name__.lower())
    ]
    module_tree = []
    for name, module in named_modules.items():
        if not name:
            continue
        depth = name.count(".") + 1
        # Numbered grandchildren of repeated block containers would make the
        # console inventory unnecessarily enormous; attention leaves are
        # reported separately below.
        if depth <= 2:
            module_tree.append({"name": name, "type": type(module).__name__, "depth": depth})

    probe_modules = [name for name in DEFAULT_PROBE_MODULES if name in named_modules]
    candidates = [
        {"module": name, "reason": reason}
        for name, reason in (
            ("blocks1.1", "last stride-4 convolutional encoder block"),
            ("blocks2.1", "last stride-8 convolutional encoder block"),
            ("patch_embed4", "stride-16 token projection before positional encoding"),
            ("blocks3.10", "last stride-16 transformer encoder block"),
            ("norm", "normalized stride-16 semantic tokens returned by UNIV"),
        )
        if name in named_modules
    ]
    return {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
        "frozen_parameter_count": sum(parameter.numel() for parameter in parameters if not parameter.requires_grad),
        "module_count": sum(1 for _ in model.modules()),
        "module_tree": module_tree,
        "patch_embedding_names": patch_embeddings,
        "patch_embedding_name": patch_embeddings[0] if patch_embeddings else None,
        "pos_embed_names": positional_embeddings,
        "pos_embed_name": positional_embeddings[0] if positional_embeddings else None,
        "encoder_block_groups": block_groups,
        "encoder_block_count": sum(block_groups.values()),
        "norm_layer_names": norms,
        "attention_module_names": attentions,
        "candidate_feature_extraction_points": candidates,
        "probe_modules": probe_modules,
        "has_attention_api": hasattr(model, "get_last_selfattention"),
        "has_encoder_api": hasattr(model, "forward_encoder"),
        "pos_embed_shape": list(model.pos_embed.shape) if hasattr(model, "pos_embed") else None,
    }


def tensor_summary(tensor: Tensor) -> dict[str, Any]:
    """Summarize an activation without serializing its potentially large data."""
    detached = tensor.detach()
    summary: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).removeprefix("torch."),
        "device": str(detached.device),
        "requires_grad": tensor.requires_grad,
        "numel": detached.numel(),
    }
    if detached.numel() and (detached.is_floating_point() or detached.is_complex()):
        finite = torch.isfinite(detached)
        summary["finite_fraction"] = finite.float().mean().item()
        if finite.any():
            values = detached[finite].float()
            summary.update(min=values.min().item(), max=values.max().item(), mean=values.mean().item())
    return summary


def _summarize_output(output: Any) -> Any:
    if isinstance(output, Tensor):
        return tensor_summary(output)
    if isinstance(output, (tuple, list)):
        return [_summarize_output(item) for item in output]
    if isinstance(output, dict):
        return {str(key): _summarize_output(value) for key, value in output.items()}
    if output is None:
        return None
    return {"type": f"{type(output).__module__}.{type(output).__qualname__}"}


@contextmanager
def capture_module_outputs(model: nn.Module, names: Sequence[str]) -> Iterator[dict[str, Any]]:
    """Capture summaries from named modules and always remove hook handles."""
    modules = dict(model.named_modules())
    unknown = [name for name in names if name not in modules]
    if unknown:
        raise ValueError(f"unknown probe modules: {', '.join(unknown)}")
    captured: dict[str, Any] = {}
    handles = []
    for name in names:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, *, name: str = name) -> None:
            captured[name] = _summarize_output(output)

        handles.append(modules[name].register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def inspect_features(
    model: nn.Module,
    *,
    batch_size: int = 1,
    image_size: tuple[int, int] = (224, 224),
    module_names: Sequence[str] = DEFAULT_PROBE_MODULES,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a synthetic no-gradient forward pass with attention enabled."""
    if batch_size < 1 or min(image_size) < 1:
        raise ValueError("batch size and image dimensions must be positive")
    model = model.to(device).eval()
    image = torch.zeros(batch_size, 3, *image_size, device=device)
    with capture_module_outputs(model, module_names) as activations, torch.inference_mode():
        output = model(image, mask_ratio=0, return_last_attention=True)
    latent, attention = output
    return {
        "input": tensor_summary(image),
        "output": {
            "latent": tensor_summary(latent),
            "attention": tensor_summary(attention) if attention is not None else None,
        },
        "activations": activations,
        "candidate_adapter_points": [
            {"module": name, "reason": reason}
            for name, reason in (
                ("blocks1.1", "last stride-4 convolutional stage"),
                ("blocks2.1", "last stride-8 convolutional stage"),
                ("norm", "normalized stride-16 UNIV semantic tokens"),
            )
            if name in activations
        ],
        "semantic_anchor": "output.latent (teacher RGB tokens in the original paired objective)",
        "attention_available": attention is not None,
    }


def inspect_checkpoint_against_model(model: nn.Module, checkpoint: str | Path) -> dict[str, Any]:
    """Load a checkpoint and expose all compatibility details as JSON data."""
    report: CheckpointLoadReport = load_univ_checkpoint(model, checkpoint)
    return asdict(report)
