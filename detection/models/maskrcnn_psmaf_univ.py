"""Mask R-CNN backbone integration boundary for PSMAF-UNIV."""

from torch import nn


class MaskRCNNPSMAFUNIVBackbone(nn.Module):
    """Return named pyramid levels in the form expected by torchvision detectors."""

    def __init__(self, psmaf_univ: nn.Module, out_channels: int = 256) -> None:
        super().__init__()
        self.body = psmaf_univ
        self.out_channels = out_channels

    def forward(self, images):
        return {str(index): feature for index, feature in enumerate(self.body(images))}
