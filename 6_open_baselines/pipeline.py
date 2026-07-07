#!/usr/bin/env python3
"""
CLI pipeline Open Baselines — ST open source (audio → texte EN).

Ce fichier est un **routeur uniquement** : il délègue aux modules sous ``6_open_baselines/``.

Usage :
    python 6_open_baselines/pipeline.py evaluate \\
        --config 6_open_baselines/configs/fr-en/seamless_m4t_v2_large.yaml \\
        --run-id run_073_open_seamlessm4t_v2_utterance --dry-run
    python 6_open_baselines/pipeline.py infer \\
        --config 6_open_baselines/configs/fr-en/whisper_large_v3_st.yaml \\
        --input-audio path/to/audio.wav --dry-run
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_open_baselines_package() -> None:
    """Enregistrer le namespace ``OpenBaselines`` depuis ``6_open_baselines/``."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts_communs.variant_bootstrap import bootstrap_open_baselines

    bootstrap_open_baselines()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Ajouter des arguments communs aux sous-commandes open baselines."""
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Router vers ``OpenBaselines.evaluate_open``."""
    _ensure_open_baselines_package()
    from OpenBaselines.evaluate_open import run_from_namespace

    return run_from_namespace(args)


def cmd_infer(args: argparse.Namespace) -> int:
    """Router vers ``OpenBaselines.infer_open``."""
    _ensure_open_baselines_package()
    from OpenBaselines.infer_open import run_from_namespace

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
        description="Pipeline Open Baselines — ST open source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_eval = subparsers.add_parser(
        "evaluate",
        help="SacreBLEU valid/test via modèle ST open",
    )
    add_common_args(p_eval)
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = subparsers.add_parser("infer", help="Inférence WAV via modèle ST open")
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
