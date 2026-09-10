"""Head-neutral PSMAF-UNIV feature extractor."""

from collections.abc import Sequence

from torch import Tensor, nn

from .multiscale_task_adapter import MultiscaleTaskAdapter
from .psmaf_fusion import PSMAFFusion
from .pseudo_semantic_guidance import PseudoSemanticGuidance
from .univ_encoder_wrapper import UNIVEncoderWrapper


class PSMAFUNIVModel(nn.Module):
    """Wrap UNIV and emit fine-to-coarse, adaptively fused task features."""

    def __init__(
        self,
        encoder: nn.Module,
        in_channels: list[int],
        out_channels: int = 256,
        *,
        allow_square_infer: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = UNIVEncoderWrapper(encoder, return_dict=True)
        self.adapter = MultiscaleTaskAdapter(
            in_channels, out_channels, allow_square_infer=allow_square_infer
        )
        self.guidance = nn.ModuleList(PseudoSemanticGuidance(out_channels) for _ in in_channels)
        self.fusion = nn.ModuleList(PSMAFFusion(out_channels) for _ in in_channels[:-1])

    def forward(
        self,
        image: Tensor,
        *,
        rgb_spatial_shape: tuple[int, int] | Sequence[tuple[int, int] | None] | None = None,
        ir_spatial_shape: tuple[int, int] | Sequence[tuple[int, int] | None] | None = None,
        modality: str = "rgb",
    ) -> tuple[Tensor, ...]:
        if modality not in {"rgb", "ir"}:
            raise ValueError("modality must be 'rgb' or 'ir'")
        spatial_shape = rgb_spatial_shape if modality == "rgb" else ir_spatial_shape
        encoded = self.encoder(image, spatial_shape=spatial_shape, modality=modality)
        adapter_features = []
        for item in encoded:
            tensor = item.get("features", item.get("tokens"))
            adapter_features.append({"tensor": tensor, "spatial_shape": item.get("spatial_shape")})
        try:
            features = list(self.adapter(adapter_features))
        except ValueError as error:
            if "require spatial_shape" in str(error):
                argument = "rgb_spatial_shape" if modality == "rgb" else "ir_spatial_shape"
                raise ValueError(
                    f"UNIV encoder returned BNC tokens without spatial_shape; include spatial_shape "
                    f"in its feature dictionary or pass {argument} to PSMAFUNIVModel.forward(). "
                    "Square-grid inference is disabled unless allow_square_infer=True."
                ) from error
            raise
        for level in range(len(features) - 2, -1, -1):
            semantic_map = self.guidance[level + 1](features[level + 1])
            features[level] = self.fusion[level](features[level], features[level + 1], semantic_map)
        return tuple(features)
