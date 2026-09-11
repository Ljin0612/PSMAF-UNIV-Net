"""Composable PSMAF-UNIV research components.

The model is imported lazily so read-only dataset utilities remain usable in
environments where the optional PyTorch runtime is unavailable.
"""

from typing import Any

__all__ = ["PSMAFUNIVModel"]


def __getattr__(name: str) -> Any:
    if name == "PSMAFUNIVModel":
        from .psmaf_univ_model import PSMAFUNIVModel

        return PSMAFUNIVModel
    raise AttributeError(name)
