"""UPerNet integration boundary for PSMAF-UNIV."""

from torch import nn


class UPerNetPSMAFUNIVBackbone(nn.Module):
    def __init__(self, psmaf_univ: nn.Module) -> None:
        super().__init__()
        self.body = psmaf_univ

    def forward(self, images):
        return self.body(images)
