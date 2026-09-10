"""Optional PEFT integration kept separate from the original UNIV model."""

from torch import nn


def attach_lora(model: nn.Module, target_modules: list[str], *, rank: int = 8, alpha: int = 32) -> nn.Module:
    """Attach LoRA adapters; PEFT is imported only when this feature is requested."""
    from peft import LoraConfig, get_peft_model

    return get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, target_modules=target_modules))
