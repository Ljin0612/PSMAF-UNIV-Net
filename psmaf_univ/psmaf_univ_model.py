"""Head-neutral PSMAF-UNIV feature extractor."""

from torch import Tensor, nn

from .multiscale_task_adapter import MultiscaleTaskAdapter
from .psmaf_fusion import PSMAFFusion
from .pseudo_semantic_guidance import PseudoSemanticGuidance
from .univ_encoder_wrapper import UNIVEncoderWrapper


class PSMAFUNIVModel(nn.Module):
    """Wrap UNIV and emit fine-to-coarse, adaptively fused task features."""

    def __init__(self, encoder: nn.Module, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        self.encoder = UNIVEncoderWrapper(encoder)
        self.adapter = MultiscaleTaskAdapter(in_channels, out_channels)
        self.guidance = nn.ModuleList(PseudoSemanticGuidance(out_channels) for _ in in_channels)
        self.fusion = nn.ModuleList(PSMAFFusion(out_channels) for _ in in_channels[:-1])

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        features = list(self.adapter(self.encoder(image)))
        for level in range(len(features) - 2, -1, -1):
            semantic_map = self.guidance[level + 1](features[level + 1])
            features[level] = self.fusion[level](features[level], features[level + 1], semantic_map)
        return tuple(features)
