#!/usr/bin/env python3
"""Discover and inspect the original, checked-in UNIV encoder implementation."""
import argparse
import ast
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psmaf_univ.univ_diagnostics import build_original_univ, model_inventory
from tools.univ_remediation import missing_module_remediation


def discover_candidates(source_root: Path) -> list[dict[str, Any]]:
    """Find likely encoder definitions without importing optional dependencies."""
    candidates: list[dict[str, Any]] = []
    if not source_root.is_dir():
        return candidates
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(source_root)
        import_path = ".".join(relative.with_suffix("").parts)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for target in targets:
                        if isinstance(target, ast.Name) and any(
                            term in target.id.lower() for term in ("convmae", "mcmae", "univ", "convvit")
                        ):
                            candidates.append({
                                "import_path": f"{import_path}.{target.id}",
                                "kind": "alias",
                                "source_file": str(relative),
                                "line": node.lineno,
                            })
                continue
            qualified = f"{import_path}.{node.name}"
            haystack = f"{node.name} {ast.get_docstring(node) or ''}".lower()
            if any(term in haystack for term in ("convmae", "mcmae", "maskedautoencoder", "univ", "convvit")):
                candidates.append({
                    "import_path": qualified,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "source_file": str(relative),
                    "line": node.lineno,
                })
    return candidates


def inspect(source_root: Path) -> dict[str, Any]:
    """Return discovery results and, when possible, a real model inventory."""
    root = source_root.resolve()
    report: dict[str, Any] = {
        "source_root": str(root),
        "import_path_candidates": discover_candidates(root),
        "selected_constructor": "models.backbone.mcmae.models_convmae.convmae_convvit_base_patch16",
        "dependency_versions": {},
    }
    for dependency in ("torch", "numpy", "timm"):
        try:
            report["dependency_versions"][dependency] = version(dependency)
        except PackageNotFoundError:
            report["dependency_versions"][dependency] = None
    try:
        model = build_original_univ(root)
    except Exception as error:  # diagnostic boundary: preserve the exact cause for users
        missing = error.name if isinstance(error, ModuleNotFoundError) else None
        numpy_compatibility = "module 'numpy' has no attribute" in str(error)
        report["construction"] = {
            "status": "unavailable",
            "error_type": type(error).__name__,
            "error": str(error),
            "missing_dependency": missing,
            "incompatible_dependency": "numpy" if numpy_compatibility else None,
            "remediation": (
                missing_module_remediation(missing)
                if missing else "Restore/fix the original UNIV source or its dependencies and rerun."
            ),
        }
        if numpy_compatibility:
            report["construction"]["remediation"] = (
                "Use a NumPy version compatible with the uploaded UNIV source "
                "(its requirements pin numpy==1.23.5), or update that source explicitly."
            )
    else:
        report["construction"] = {"status": "ok"}
        report["model"] = model_inventory(model)
    return report


def print_human(report: dict[str, Any]) -> None:
    print("UNIV original encoder inspection")
    print("=" * 32)
    print(f"Source root: {report['source_root']}")
    print("\nImport path candidates:")
    for item in report["import_path_candidates"]:
        print(f"  - {item['import_path']} ({item['kind']}, {item['source_file']}:{item['line']})")
    if not report["import_path_candidates"]:
        print("  (none discovered)")
    print(f"Selected constructor: {report['selected_constructor']}")
    versions = ", ".join(f"{name}={value or 'not installed'}" for name, value in report["dependency_versions"].items())
    print(f"Dependency versions: {versions}")
    construction = report["construction"]
    print(f"Construction: {construction['status']}")
    if construction["status"] != "ok":
        print(f"ERROR: {construction['error_type']}: {construction['error']}")
        if construction.get("missing_dependency"):
            print(f"Missing dependency: {construction['missing_dependency']}")
        if construction.get("incompatible_dependency"):
            print(f"Incompatible dependency: {construction['incompatible_dependency']}")
        print(f"Next step: {construction['remediation']}")
        return
    model = report["model"]
    print(f"\nModel class: {model['class']}")
    print(f"Parameters: {model['parameter_count']:,} total; {model['trainable_parameter_count']:,} trainable; {model['frozen_parameter_count']:,} frozen")
    print(f"Patch embedding module: {model['patch_embedding_name'] or '(not found)'}")
    print(f"Patch embedding modules: {', '.join(model['patch_embedding_names']) or '(none)'}")
    print(f"pos_embed parameter: {model['pos_embed_name'] or '(not found)'}")
    print(f"Encoder blocks: {model['encoder_block_count']} ({model['encoder_block_groups']})")
    print(f"Norm layers: {', '.join(model['norm_layer_names']) or '(none)'}")
    print(f"Attention modules: {', '.join(model['attention_module_names']) or '(none discovered)'}")
    print("\nCandidate feature extraction points:")
    for item in model["candidate_feature_extraction_points"]:
        print(f"  - {item['module']}: {item['reason']}")
    print("\nModule tree summary:")
    for item in model["module_tree"]:
        print(f"  {'  ' * (item['depth'] - 1)}{item['name']} ({item['type']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "UNIV-main")
    parser.add_argument("--json-out", type=Path, help="also write the complete report as JSON")
    args = parser.parse_args()
    report = inspect(args.source_root)
    print_human(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report written to {args.json_out}")


if __name__ == "__main__":
    main()
