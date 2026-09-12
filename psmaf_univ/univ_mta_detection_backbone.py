"""Single-stream UNIV + MTA backbone for torchvision detectors."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .checkpoint_loader import load_univ_checkpoint
from .multiscale_task_adapter import MultiScaleTaskAdapter
from .univ_diagnostics import build_original_univ


def validate_checkpoint_load(report: Any, min_load_fraction: float) -> None:
    if not 0 <= min_load_fraction <= 1:
        raise ValueError("min_load_fraction must be between 0 and 1")
    get = report.get if isinstance(report, Mapping) else lambda key, default=None: getattr(report, key, default)
    fraction, count = float(get("load_fraction", 0)), int(get("loaded_key_count", 0))
    if count == 0 or fraction < min_load_fraction:
        raise RuntimeError(
            f"UNIV checkpoint load_fraction {fraction:.6g} is below required minimum "
            f"{min_load_fraction:.6g}, or no weights were loaded"
        )


class UNIVMTADetectionBackbone(nn.Module):
    """Expose UNIV ``blocks2.1`` and ``norm`` as an MTA feature pyramid."""

    out_channels = 256

    def __init__(self, encoder: nn.Module, *, freeze_univ: bool = True) -> None:
        super().__init__()
        modules = dict(encoder.named_modules())
        missing = [name for name in ("blocks2.1", "norm") if name not in modules]
        if missing:
            raise ValueError(f"UNIV model is missing required feature modules: {', '.join(missing)}")
        self.encoder = encoder
        self.freeze_univ = freeze_univ
        if freeze_univ:
            self.encoder.requires_grad_(False)
        spatial_channels = _module_channels(modules["blocks2.1"], 384)
        semantic_channels = _module_channels(modules["norm"], 768)
        self.adapter = MultiScaleTaskAdapter(spatial_channels, semantic_channels, self.out_channels)
        self._feature_modules = (modules["blocks2.1"], modules["norm"])

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, *, checkpoint_key: str = "student",
        min_load_fraction: float = 0.5, source_root: str | Path | None = None,
        freeze_univ: bool = True,
    ) -> tuple["UNIVMTADetectionBackbone", Any]:
        encoder = build_original_univ(source_root)
        report = load_univ_checkpoint(encoder, checkpoint, checkpoint_key=checkpoint_key)
        validate_checkpoint_load(report, min_load_fraction)
        return cls(encoder, freeze_univ=freeze_univ), report

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_univ:
            self.encoder.eval()
        return self

    def forward(self, image: Tensor) -> OrderedDict[str, Tensor]:
        if tuple(image.shape[-2:]) != (224, 224):
            raise ValueError(f"UNIV detection backbone requires 224x224 input; got {tuple(image.shape[-2:])}")
        captured: dict[str, Tensor] = {}
        handles = [
            module.register_forward_hook(lambda _m, _i, output, name=name: captured.__setitem__(name, output))
            for name, module in zip(("blocks2.1", "norm"), self._feature_modules)
        ]
        try:
            context = torch.no_grad() if self.freeze_univ else torch.enable_grad()
            with context:
                output = self.encoder(image, mask_ratio=0, return_last_attention=True)
        finally:
            for handle in handles:
                handle.remove()
        semantic = captured.get("norm")
        if semantic is None and isinstance(output, (tuple, list)):
            semantic = output[0]
        if semantic is None or semantic.ndim != 3 or semantic.shape[1] != 196:
            shape = None if semantic is None else tuple(semantic.shape)
            raise ValueError(f"norm/output.latent must be BNC with grid_size=(14, 14); got {shape}")
        pyramid = self.adapter(captured["blocks2.1"], semantic, grid_size=(14, 14))
        return OrderedDict((name, pyramid[name]) for name in ("P3", "P4", "P5"))


def _module_channels(module: nn.Module, default: int) -> int:
    for attribute in ("normalized_shape", "num_features", "out_channels", "dim"):
        value = getattr(module, attribute, None)
        if isinstance(value, (tuple, list)):
            value = value[-1]
        if isinstance(value, int):
            return value
    return default
