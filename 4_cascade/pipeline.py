#!/usr/bin/env python3
"""
CLI pipeline Cascade — baseline ST ASR→MT (audio → texte FR → texte EN).

Ce fichier est un **routeur uniquement** : il délègue aux modules sous ``4_cascade/``.

Usage :
    python 4_cascade/pipeline.py evaluate --config 4_cascade/configs/fr-en/cascade.yaml \\
        --run-id run_001_cascade --dry-run
    python 4_cascade/pipeline.py infer --config 4_cascade/configs/fr-en/cascade.yaml \\
        --input-audio path/to/audio.wav --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_cascade_package() -> None:
    """Enregistrer le namespace ``Cascade`` depuis ``4_cascade/``."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts_communs.variant_bootstrap import bootstrap_cascade

    bootstrap_cascade()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Ajouter des arguments communs aux sous-commandes Cascade."""
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Router vers ``Cascade.evaluate_cascade``."""
    _ensure_cascade_package()
    from Cascade.evaluate_cascade import run_from_namespace

    return run_from_namespace(args)


def cmd_infer(args: argparse.Namespace) -> int:
    """Router vers ``Cascade.infer_cascade``."""
    _ensure_cascade_package()
    from Cascade.infer_cascade import run_from_namespace

    return run_from_namespace(args)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Construire les sous-parseurs et dispatcher vers la sous-commande choisie.

    Paramètres :
        argv : Arguments CLI (``None`` = ``sys.argv``).

    Retour :
        Code de sortie du stage invoqué.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline Cascade — baseline ASR→MT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eval = subparsers.add_parser(
        "evaluate",
        help="SacreBLEU valid/test via ASR puis MT",
    )
    add_common_args(p_eval)
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = subparsers.add_parser("infer", help="Inférence WAV via ASR→MT")
    add_common_args(p_infer)
    p_infer.add_argument("--input-audio", type=Path, required=True)
    p_infer.add_argument("--config", type=Path, default=None)
    p_infer.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    p_infer.set_defaults(func=cmd_infer)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
