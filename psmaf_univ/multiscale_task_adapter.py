"""Projection of heterogeneous backbone levels into task-ready feature maps."""

from collections.abc import Sequence
from typing import cast

from torch import Tensor, nn


def _normalize_shape_alias(
    shape: tuple[int, int] | Sequence[tuple[int, int] | None] | None,
    num_levels: int,
) -> list[tuple[int, int] | None]:
    """Normalize a shape alias to one optional shape per adapter level."""
    if shape is None:
        return [None] * num_levels
    if num_levels == 1 and len(shape) == 2 and all(isinstance(value, int) for value in shape):
        return [cast(tuple[int, int], tuple(shape))]

    shapes = list(shape)  # type: ignore[arg-type]
    if len(shapes) != num_levels:
        raise ValueError("spatial shape count does not match adapter levels")
    return [None if value is None else tuple(value) for value in shapes]


class MultiscaleTaskAdapter(nn.Module):
    """Project BCHW maps or BNC tokens to a shared channel width.

    BNC inputs require an explicit ``spatial_shape`` (or its ``grid_size`` alias)
    to preserve rectangular grids. For a single level, both ``(height, width)``
    and ``[(height, width)]`` are accepted. Aliases are normalized to per-level
    shapes before conflicts are checked. A feature may instead be
    ``{"tensor": tokens, "spatial_shape": (height, width)}``. Legacy square
    inference is available only when ``allow_square_infer=True``.
    """

    def __init__(self, in_channels: list[int], out_channels: int, *, allow_square_infer: bool = False) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(channels, out_channels, 1) for channels in in_channels)
        self.allow_square_infer = allow_square_infer

    def forward(
        self,
        features: Tensor | Sequence[Tensor | dict],
        spatial_shape: tuple[int, int] | Sequence[tuple[int, int] | None] | None = None,
        *,
        grid_size: tuple[int, int] | Sequence[tuple[int, int] | None] | None = None,
        allow_square_infer: bool | None = None,
    ) -> tuple[Tensor, ...]:
        """Project features, validating every explicit BNC token-grid shape."""
        feature_list = [features] if isinstance(features, (Tensor, dict)) else list(features)
        if len(feature_list) != len(self.projections):
            raise ValueError("feature count does not match configured adapter levels")

        spatial_shapes = _normalize_shape_alias(spatial_shape, len(feature_list))
        grid_shapes = _normalize_shape_alias(grid_size, len(feature_list))
        if spatial_shape is not None and grid_size is not None and spatial_shapes != grid_shapes:
            raise ValueError("spatial_shape and grid_size must match when both are provided")
        shape_list = spatial_shapes if spatial_shape is not None else grid_shapes

        infer_square = self.allow_square_infer if allow_square_infer is None else allow_square_infer
        maps = []
        for item, supplied_shape, projection in zip(feature_list, shape_list, self.projections):
            feature = item
            if isinstance(item, dict):
                feature = item.get(
                    "tensor", item.get("feature", item.get("features", item.get("tokens")))
                )
                item_spatial_shape = item.get("spatial_shape")
                item_grid_size = item.get("grid_size")
                item_spatial_shapes = _normalize_shape_alias(item_spatial_shape, 1)
                item_grid_sizes = _normalize_shape_alias(item_grid_size, 1)
                if item_spatial_shape is not None and item_grid_size is not None and (
                    item_spatial_shapes != item_grid_sizes
                ):
                    raise ValueError("spatial_shape and grid_size must match when both are provided")
                dict_shape = (
                    item_spatial_shapes[0] if item_spatial_shape is not None else item_grid_sizes[0]
                )
                if supplied_shape is not None and dict_shape is not None:
                    raise ValueError("spatial shape was provided both separately and in the feature dict")
                supplied_shape = dict_shape if dict_shape is not None else supplied_shape
            if not isinstance(feature, Tensor):
                raise ValueError("feature dict must contain a Tensor under 'tensor' or 'feature'")
            if feature.ndim == 3:
                if supplied_shape is None:
                    if not infer_square:
                        raise ValueError(
                            "BNC inputs require spatial_shape=(height, width) to preserve rectangular grids; "
                            "set allow_square_infer=True only for known-square grids"
                        )
                    side = int(feature.shape[1] ** 0.5)
                    if side * side != feature.shape[1]:
                        raise ValueError("cannot infer a square grid from the BNC token count")
                    supplied_shape = (side, side)
                height, width = supplied_shape
                if height * width != feature.shape[1]:
                    raise ValueError(
                        f"BNC token count ({feature.shape[1]}) does not match spatial shape "
                        f"({height}, {width})"
                    )
                feature = feature.transpose(1, 2).reshape(feature.shape[0], feature.shape[2], height, width)
            if feature.ndim != 4:
                raise ValueError("features must be BCHW maps or BNC token tensors")
            maps.append(projection(feature))
        return tuple(maps)
