#!/usr/bin/env python3
"""
Enregistrement de namespaces Python pour dossiers à nom non importable (préfixe numérique).

Les variantes vivent sous ``2_speechLLM/``, ``3_Gemini/``, ``4_cascade/`` mais le code
historique importe ``speechLLM.*``, ``Gemini.*``, ``Cascade.*``. Ce module charge les
fichiers ``.py`` du répertoire cible dans ``sys.modules`` sous l'alias demandé.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def register_variant_package(alias: str, directory: Path) -> None:
    """
    Exposer les modules ``*.py`` d'un répertoire variant sous un nom de package logique.

    Paramètres :
        alias : Nom importable (ex. ``speechLLM``, ``Gemini``).
        directory : Dossier racine de la variante (ex. ``2_speechLLM/``).

    Effet :
        Renseigne ``sys.modules[alias.<module>]`` pour chaque script sibling (hors
        ``pipeline.py`` et fichiers privés ``_*.py``).
    """
    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Variant directory not found: {root}")

    pkg = sys.modules.get(alias)
    if pkg is None:
        pkg = types.ModuleType(alias)
        pkg.__path__ = [str(root)]  # type: ignore[attr-defined]
        sys.modules[alias] = pkg
    elif not hasattr(pkg, "__path__"):
        pkg.__path__ = [str(root)]  # type: ignore[attr-defined]

    skip = {"pipeline.py"}
    for py_file in sorted(root.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name in skip:
            continue
        mod_name = f"{alias}.{py_file.stem}"
        if mod_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(mod_name, py_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {py_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)


def bootstrap_speechllm() -> None:
    """Enregistrer le package logique ``speechLLM`` depuis ``2_speechLLM/``."""
    register_variant_package("speechLLM", PROJECT_ROOT / "2_speechLLM")


def bootstrap_gemini() -> None:
    """Enregistrer ``Gemini`` (dépend de ``speechLLM`` pour les utilitaires communs)."""
    bootstrap_speechllm()
    register_variant_package("Gemini", PROJECT_ROOT / "3_Gemini")


def bootstrap_cascade() -> None:
    """Enregistrer ``Cascade`` (dépend de ``speechLLM`` pour les utilitaires communs)."""
    bootstrap_speechllm()
    register_variant_package("Cascade", PROJECT_ROOT / "4_cascade")
