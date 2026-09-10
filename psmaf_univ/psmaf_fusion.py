"""Pseudo-semantic guided fusion across adjacent feature scales."""

import torch.nn.functional as F
from torch import Tensor, nn


class PSMAFFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feature: Tensor, context: Tensor, guidance: Tensor) -> Tensor:
        context = F.interpolate(context, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        guidance = F.interpolate(guidance, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(feature + guidance * context)
