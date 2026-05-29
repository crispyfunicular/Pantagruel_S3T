"""Shared pytest fixtures for S3T."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def pytest_configure() -> None:
    """Enregistrer les packages logiques des variantes à nom préfixé numérique."""
    from scripts_communs.variant_bootstrap import (
        bootstrap_cascade,
        bootstrap_gemini,
        bootstrap_speechllm,
    )

    bootstrap_speechllm()
    bootstrap_gemini()
    bootstrap_cascade()


def load_stage_module(filename: str, *, root: Path | None = None):
    """Load a numbered stage script (e.g. 0_preflight.py) as a module."""
    base = root or (PROJECT_ROOT / "scripts_communs")
    path = base / filename
    spec = importlib.util.spec_from_file_location(
        filename.replace(".py", "").replace("-", "_"),
        path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
