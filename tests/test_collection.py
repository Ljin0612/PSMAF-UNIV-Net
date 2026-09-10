"""Checks that remain runnable when optional machine-learning packages are absent."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_project_python_sources_parse_without_importing_optional_dependencies():
    """Keep at least one useful test runnable when the torch suite is skipped."""
    paths = [
        *sorted((ROOT / "psmaf_univ").glob("*.py")),
        *sorted((ROOT / "tools").glob("*.py")),
    ]
    assert paths
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
