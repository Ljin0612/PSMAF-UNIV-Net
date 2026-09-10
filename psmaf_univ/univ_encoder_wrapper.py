"""Adapter around a UNIV-compatible encoder without modifying its source."""

from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn


class UNIVEncoderWrapper(nn.Module):
    """Normalize encoder outputs while retaining token layout metadata.

    ``return_dict=False`` preserves the original tuple-of-tensors interface.
    Structured mode represents each encoder level with a dictionary so BNC token
    grids do not lose their spatial shape on their way to a task adapter.
    """

    def __init__(self, encoder: nn.Module, *, return_dict: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.return_dict = return_dict

    @staticmethod
    def _as_dict(
        value: Tensor | dict[str, Any],
        *,
        spatial_shape: tuple[int, int] | None,
        modality: str,
    ) -> dict[str, Any]:
        if isinstance(value, Tensor):
            result: dict[str, Any] = {"tokens" if value.ndim == 3 else "features": value}
        elif isinstance(value, dict):
            result = dict(value)
            # Accept adapter-style encoder dictionaries as well as the public names.
            tensor = result.pop("tensor", result.pop("feature", None))
            if tensor is not None and "tokens" not in result and "features" not in result:
                result["tokens" if tensor.ndim == 3 else "features"] = tensor
        else:
            raise TypeError("UNIV encoder features must be tensors or dictionaries")

        if not isinstance(result.get("tokens", result.get("features")), Tensor):
            raise TypeError("UNIV encoder feature dictionaries must contain a tensor")
        if spatial_shape is not None:
            if result.get("spatial_shape") is not None and tuple(result["spatial_shape"]) != spatial_shape:
                raise ValueError("encoder and caller supplied different spatial_shape values")
            result["spatial_shape"] = spatial_shape
        result.setdefault("modality", modality)
        result["source"] = "univ_encoder"
        result.setdefault("debug", {})
        return result

    def forward(
        self,
        image: Tensor,
        *,
        spatial_shape: tuple[int, int] | Sequence[tuple[int, int] | None] | None = None,
        modality: str = "rgb",
        return_dict: bool | None = None,
    ) -> tuple[Tensor, ...] | tuple[dict[str, Any], ...]:
        if modality not in {"rgb", "ir"}:
            raise ValueError("modality must be 'rgb' or 'ir'")
        output = self.encoder(image)
        if (
            isinstance(output, dict)
            and isinstance(output.get("features"), Sequence)
            and not isinstance(output.get("features"), Tensor)
        ):
            values = list(output["features"])
        elif isinstance(output, dict) and not any(
            key in output for key in ("tokens", "features", "tensor", "feature")
        ):
            values: list[Tensor | dict[str, Any]] = list(output.values())
        elif isinstance(output, (Tensor, dict)):
            values = [output]
        elif isinstance(output, Sequence):
            values = list(output)
        else:
            raise TypeError("UNIV encoder must return a tensor, dictionary, or sequence")

        if spatial_shape is None:
            shapes = [None] * len(values)
        elif (
            len(values) == 1
            and len(spatial_shape) == 2
            and all(isinstance(x, int) for x in spatial_shape)
        ):
            shapes = [tuple(spatial_shape)]
        else:
            shapes = list(spatial_shape)  # type: ignore[arg-type]
            if len(shapes) != len(values):
                raise ValueError("spatial shape count does not match encoder feature levels")

        structured = tuple(
            self._as_dict(value, spatial_shape=shape, modality=modality)
            for value, shape in zip(values, shapes)
        )
        use_dict = self.return_dict if return_dict is None else return_dict
        if use_dict:
            return structured
        return tuple(item.get("features", item.get("tokens")) for item in structured)
