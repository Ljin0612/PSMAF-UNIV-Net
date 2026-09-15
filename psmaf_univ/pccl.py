"""Patch-level cross-modal contrastive learning used by Stage 7."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def attention_pseudo_labels(attention: Tensor, gamma: float = 0.6) -> Tensor:
    """Turn frozen RGB self-attention into a binary patch-pair matrix.

    UNIV attention is ``B,H,N,N``. Heads are averaged and the directed map is
    symmetrised. Values are normalized per image before thresholding so gamma
    has resolution-independent meaning. Self pairs are always positives.
    """
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if attention.ndim == 4:
        attention = attention.mean(dim=1)
    if attention.ndim != 3 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError("attention must have shape BxNxN or BxHxNxN")
    scores = (attention + attention.transpose(-1, -2)) * 0.5
    minimum = scores.amin(dim=(-2, -1), keepdim=True)
    maximum = scores.amax(dim=(-2, -1), keepdim=True)
    scores = (scores - minimum) / (maximum - minimum).clamp_min(torch.finfo(scores.dtype).eps)
    labels = (scores >= gamma).to(dtype=scores.dtype)
    diagonal = torch.arange(labels.shape[-1], device=labels.device)
    labels[:, diagonal, diagonal] = 1
    return labels.detach()


def sample_patch_indices(token_count: int, num_patches: int, device=None, generator=None) -> Tensor:
    """Return a shared random subset, avoiding a full 1600-square PCCL matrix."""
    if token_count < 1 or num_patches < 1:
        raise ValueError("token_count and num_patches must be positive")
    count = min(token_count, num_patches)
    return torch.randperm(token_count, device=device, generator=generator)[:count]


class PCCLLoss(nn.Module):
    """Sigmoid-similarity BCE for frozen anchors and aligned modality patches."""

    def __init__(self, temperature: float = 0.04) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, anchors: Tensor, patches: Tensor, pseudo_labels: Tensor) -> Tensor:
        if anchors.shape != patches.shape or anchors.ndim != 3:
            raise ValueError("anchors and patches must have matching BxNxC shape")
        expected = (*anchors.shape[:2], anchors.shape[1])
        if tuple(pseudo_labels.shape) != expected:
            raise ValueError(f"pseudo_labels must have shape {expected}")
        anchors = F.normalize(anchors.detach(), dim=-1)
        patches = F.normalize(patches, dim=-1)
        logits = torch.bmm(anchors, patches.transpose(1, 2)) / self.temperature
        return F.binary_cross_entropy_with_logits(logits, pseudo_labels.to(logits.dtype))


def pccl_objective(
    anchors: Tensor, ir_patches: Tensor, rgb_patches: Tensor | None,
    pseudo_labels: Tensor, *, alpha: float = 1.0, beta: float = 1.0,
    temperature: float = 0.04,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return weighted PCCL, IR-anchor, and visible-anchor losses."""
    criterion = PCCLLoss(temperature)
    loss_ia = criterion(anchors, ir_patches, pseudo_labels)
    loss_va = criterion(anchors, rgb_patches, pseudo_labels) if rgb_patches is not None else loss_ia.new_zeros(())
    return alpha * loss_ia + beta * loss_va, loss_ia, loss_va
