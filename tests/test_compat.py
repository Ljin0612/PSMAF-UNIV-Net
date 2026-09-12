"""Tests for compatibility with the checked-in original UNIV source."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

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
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), old_pythonpath) if part
    )
    code = textwrap.dedent(
        """
        from psmaf_univ.compat import apply_numpy_legacy_aliases
        import numpy as np

        apply_numpy_legacy_aliases()
        apply_numpy_legacy_aliases()

        assert hasattr(np, "float")
        assert hasattr(np, "int")
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    assert result.returncode == 0
