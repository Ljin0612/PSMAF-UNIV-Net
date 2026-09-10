import torch
from torch import nn

from psmaf_univ import PSMAFUNIVModel
from psmaf_univ.checkpoint_loader import extract_state_dict


class Encoder(nn.Module):
    def forward(self, image):
        return (image, nn.functional.avg_pool2d(image, 2), nn.functional.avg_pool2d(image, 4))


def test_model_emits_multiscale_features():
    model = PSMAFUNIVModel(Encoder(), [3, 3, 3], out_channels=8)
    output = model(torch.randn(2, 3, 32, 32))
    assert [item.shape for item in output] == [(2, 8, 32, 32), (2, 8, 16, 16), (2, 8, 8, 8)]


def test_checkpoint_branch_priority():
    tensor = torch.ones(1)
    assert extract_state_dict({"student": {"weight": tensor}})["weight"] is tensor
