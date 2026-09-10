"""Projection of heterogeneous backbone levels into task-ready feature maps."""

from torch import Tensor, nn


class MultiscaleTaskAdapter(nn.Module):
    def __init__(self, in_channels: list[int], out_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(channels, out_channels, 1) for channels in in_channels)

    def forward(self, features: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if len(features) != len(self.projections):
            raise ValueError("feature count does not match configured adapter levels")
        maps = []
        for feature, projection in zip(features, self.projections):
            if feature.ndim == 3:
                side = int(feature.shape[1] ** 0.5)
                if side * side != feature.shape[1]:
                    raise ValueError("token count must form a square feature map")
                feature = feature.transpose(1, 2).reshape(feature.shape[0], feature.shape[2], side, side)
            if feature.ndim != 4:
                raise ValueError("features must be BCHW maps or BNC token tensors")
            maps.append(projection(feature))
        return tuple(maps)
