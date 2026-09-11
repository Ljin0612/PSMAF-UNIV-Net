"""Tests for compatibility with the checked-in original UNIV source."""

import numpy as np

from psmaf_univ.compat import apply_numpy_legacy_aliases


def test_apply_numpy_legacy_aliases_restores_missing_float(monkeypatch):
    monkeypatch.delattr(np, "float", raising=False)

    apply_numpy_legacy_aliases()

    assert np.float is float


def test_apply_numpy_legacy_aliases_is_idempotent(monkeypatch):
    monkeypatch.delattr(np, "float", raising=False)
    monkeypatch.delattr(np, "int", raising=False)

    apply_numpy_legacy_aliases()
    first_float = np.float
    first_int = np.int
    apply_numpy_legacy_aliases()

    assert np.float is first_float
    assert np.int is first_int
