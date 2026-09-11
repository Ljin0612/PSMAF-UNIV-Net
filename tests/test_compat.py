"""Tests for compatibility with the checked-in original UNIV source."""

from types import SimpleNamespace

import numpy as np

from psmaf_univ import compat


def test_apply_numpy_legacy_aliases_restores_missing_float(monkeypatch):
    fake_np = SimpleNamespace()
    monkeypatch.setattr(compat, "np", fake_np)

    compat.apply_numpy_legacy_aliases()

    assert fake_np.float is float
    assert fake_np.int is int


def test_apply_numpy_legacy_aliases_is_idempotent(monkeypatch):
    fake_np = SimpleNamespace()
    monkeypatch.setattr(compat, "np", fake_np)

    compat.apply_numpy_legacy_aliases()
    first_float = fake_np.float
    first_int = fake_np.int
    compat.apply_numpy_legacy_aliases()

    assert fake_np.float is first_float
    assert fake_np.int is first_int


def test_apply_numpy_legacy_aliases_with_real_numpy_is_safe():
    compat.apply_numpy_legacy_aliases()
    compat.apply_numpy_legacy_aliases()

    assert np.array([1, 2, 3]).sum() == 6
