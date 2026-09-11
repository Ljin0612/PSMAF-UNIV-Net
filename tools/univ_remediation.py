"""Remediation messages shared by the UNIV diagnostic tools."""

INTERNAL_UNIV_MODULES = {"models", "datasets", "loss", "utils", "SEG", "configs"}


def missing_module_remediation(name: str) -> str:
    """Explain whether a missing import belongs to UNIV or a dependency."""
    root_module = name.split(".", 1)[0]
    if root_module in INTERNAL_UNIV_MODULES:
        return (
            "Restore or fix the original UNIV source tree. The missing module appears "
            "to be part of the checked-in UNIV source, not an installable dependency."
        )
    return f"Install the unavailable dependency '{name}' and rerun."
