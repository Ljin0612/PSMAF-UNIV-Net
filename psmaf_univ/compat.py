"""Compatibility shims required by the checked-in original UNIV source."""

from __future__ import annotations

import numpy as np


def apply_numpy_legacy_aliases() -> None:
    """Restore NumPy scalar aliases used by the original UNIV snapshot.

    NumPy 1.24 removed aliases that had previously been deprecated.  Keep this
    shim deliberately narrow: it only restores aliases referenced by legacy
    model code and does not suppress unrelated import or construction errors.
    """
    namespace = vars(np)
    if "float" not in namespace:
        np.float = float  # type: ignore[attr-defined]
    if "int" not in namespace:
        np.int = int  # type: ignore[attr-defined]
