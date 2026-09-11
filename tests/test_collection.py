"""Checks that remain runnable when optional machine-learning packages are absent."""

import ast
from pathlib import Path

import pytest

from tools.univ_remediation import missing_module_remediation


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


@pytest.mark.parametrize(
    "name",
    ["models.backbone", "datasets.loader", "loss", "utils.misc", "SEG.config", "configs.base"],
)
def test_inspection_tool_identifies_missing_internal_univ_modules(name):
    assert missing_module_remediation(name) == (
        "Restore or fix the original UNIV source tree. The missing module appears "
        "to be part of the checked-in UNIV source, not an installable dependency."
    )


@pytest.mark.parametrize("name", ["torch", "timm", "peft", "transformers", "mmcv", "mmengine"])
def test_inspection_tool_identifies_missing_external_dependencies(name):
    assert missing_module_remediation(name) == (
        f"Install the unavailable dependency '{name}' and rerun."
    )
