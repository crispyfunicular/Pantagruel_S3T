#!/usr/bin/env python3
"""
CLI pipeline Gemini — baseline ST (audio → texte) via API.

Ce fichier est un **routeur uniquement** : il délègue aux modules sous `3_Gemini/`.

Usage :
    python 3_Gemini/pipeline.py evaluate --config 3_Gemini/configs/fr-en/gemini_flash.yaml --run-id run_001_gemini_flash
    python 3_Gemini/pipeline.py infer --config 3_Gemini/configs/fr-en/gemini_flash.yaml --input-audio audio.wav
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_gemini_package() -> None:
    """Enregistrer le namespace ``Gemini`` depuis ``3_Gemini/``."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts_communs.variant_bootstrap import bootstrap_gemini

    bootstrap_gemini()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Ajouter des arguments communs aux sous-commandes Gemini."""

    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Router vers `Gemini.evaluate_gemini`."""
    _ensure_gemini_package()
    from Gemini.evaluate_gemini import run_from_namespace

    return run_from_namespace(args)


def cmd_infer(args: argparse.Namespace) -> int:
    """Router vers `Gemini.infer_gemini`."""
    _ensure_gemini_package()
    from Gemini.infer_gemini import run_from_namespace

    return run_from_namespace(args)


def main(argv: Sequence[str] | None = None) -> int:
    """Construire les sous-parseurs et dispatcher vers la sous-commande choisie."""

    parser = argparse.ArgumentParser(
        description="Pipeline Gemini — baseline ST via API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eval = subparsers.add_parser("evaluate", help="SacreBLEU valid/test via Gemini")
    add_common_args(p_eval)
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.add_argument("--max-retries", type=int, default=2)
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = subparsers.add_parser("infer", help="Inférence WAV via Gemini")
    add_common_args(p_infer)
    p_infer.add_argument("--input-audio", type=Path, required=True)
    p_infer.add_argument("--config", type=Path, default=None)
    p_infer.add_argument("--model-id", type=str, default=None)
    p_infer.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    p_infer.add_argument("--max-retries", type=int, default=2)
    p_infer.set_defaults(func=cmd_infer)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
