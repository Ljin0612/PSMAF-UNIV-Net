"""Pseudo-semantic confidence estimation from UNIV task features."""

from torch import Tensor, nn


class PseudoSemanticGuidance(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.predictor = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels), nn.Conv2d(channels, 1, 1))

    def forward(self, feature: Tensor) -> Tensor:
        return self.predictor(feature).sigmoid()
