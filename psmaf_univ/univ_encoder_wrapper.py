"""Adapter around a UNIV-compatible encoder without modifying its source."""

from collections.abc import Sequence

from torch import Tensor, nn


class UNIVEncoderWrapper(nn.Module):
    """Expose an encoder output as an ordered tuple of feature tensors."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        output = self.encoder(image)
        if isinstance(output, Tensor):
            return (output,)
        if isinstance(output, dict):
            output = output.get("features", tuple(output.values()))
        if not isinstance(output, Sequence) or not all(isinstance(x, Tensor) for x in output):
            raise TypeError("UNIV encoder must return a tensor or a sequence of tensors")
        return tuple(output)
