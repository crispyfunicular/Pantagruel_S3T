#!/usr/bin/env python3
"""
CLI pipeline commun S3T — étapes 0 à 2 (données m-TEDx).

Routeur uniquement : délègue aux modules sous ``scripts_communs/``.
Les variantes numérotées (``1_Transformer``, ``2_speechLLM``, …) consomment
les manifests produits ici.

Usage :
    python scripts_communs/pipeline.py preflight
    python scripts_communs/pipeline.py download --langpairs fr-en
    python scripts_communs/pipeline.py prepare --langpair fr-en
    python scripts_communs/pipeline.py run --langpair fr-en --from-stage download --to-stage prepare
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_COMMUNS = PROJECT_ROOT / "scripts_communs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGES = ("preflight", "download", "prepare")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attacher ``--verbose``, ``--dry-run`` et ``--log-file`` à un sous-parseur."""
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Chemin optionnel du fichier journal.",
    )


def _load_stage_module(filename: str, module_label: str):
    """Importer un script numéroté depuis ``scripts_communs/``."""
    path = SCRIPTS_COMMUNS / filename
    spec = importlib.util.spec_from_file_location(module_label, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_preflight(args: argparse.Namespace) -> int:
    """Exécuter les vérifications préalables étape 0."""
    return _load_stage_module("0_preflight.py", "s3t_preflight").run_from_namespace(
        args
    )


def cmd_download(args: argparse.Namespace) -> int:
    """Exécuter le téléchargement m-TEDx étape 1."""
    return _load_stage_module("1_download.py", "s3t_download").run_from_namespace(args)


def cmd_prepare(args: argparse.Namespace) -> int:
    """Exécuter la préparation audio/manifests étape 2."""
    return _load_stage_module("2_prepare.py", "s3t_prepare").run_from_namespace(args)


def _run_stage(
    name: str, handler: Callable[[argparse.Namespace], int], args: argparse.Namespace
) -> int:
    """Afficher la bannière d'étape et invoquer le handler sauf en ``--dry-run``."""
    if args.dry_run:
        print(f"[dry-run] would run stage: {name}")
        return 0
    print(f"\n{'─' * 60}")
    print(f"Stage: {name.upper()}")
    print(f"{'─' * 60}")
    return handler(args)


def cmd_run(args: argparse.Namespace) -> int:
    """Orchestrer les étapes communes de ``--from-stage`` à ``--to-stage``."""
    from_idx = STAGES.index(args.from_stage)
    to_idx = STAGES.index(args.to_stage)
    if from_idx > to_idx:
        print(
            f"ERROR: --from-stage ({args.from_stage}) must come before "
            f"--to-stage ({args.to_stage})",
            file=sys.stderr,
        )
        return 2

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "preflight": cmd_preflight,
        "download": cmd_download,
        "prepare": cmd_prepare,
    }
    selected = STAGES[from_idx : to_idx + 1]
    print("=" * 60)
    print("S3T PIPELINE — scripts communs")
    print(f"  langpair: {getattr(args, 'langpair', '(n/a)')}")
    print(f"  stages:   {' → '.join(selected)}")
    print("=" * 60)

    for stage in selected:
        code = _run_stage(stage, handlers[stage], args)
        if code != 0:
            return code

    print("\n" + "=" * 60)
    print("PIPELINE COMMUN COMPLETE")
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Construire les sous-parseurs et dispatcher."""
    parser = argparse.ArgumentParser(
        description="Pipeline S3T — étapes communes (0–2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_preflight = subparsers.add_parser(
        "preflight", help="Valider l'environnement et les ressources"
    )
    add_common_args(p_preflight)
    p_preflight.add_argument("--min-disk-gb", type=int, default=200)
    p_preflight.add_argument("--min-vram-gb", type=int, default=8)
    p_preflight.add_argument(
        "--check-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_preflight.add_argument(
        "--check-network",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_preflight.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "preflight_report.json",
    )
    p_preflight.add_argument("--disk-path", type=Path, default=PROJECT_ROOT)
    p_preflight.add_argument("--network-timeout", type=float, default=10.0)
    p_preflight.set_defaults(func=cmd_preflight)

    p_download = subparsers.add_parser(
        "download", help="Télécharger jeux m-TEDx (OpenSLR-100)"
    )
    add_common_args(p_download)
    p_download.add_argument("--langpairs", default="fr-en")
    p_download.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "datasets" / "raw"
    )
    p_download.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    p_download.add_argument(
        "--extract", action=argparse.BooleanOptionalAction, default=True
    )
    p_download.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "download_manifest.json",
    )
    p_download.set_defaults(func=cmd_download)

    # Préparer : pour toutes les options (segment-mode, …), préférer
    # ``python scripts_communs/2_prepare.py`` ou étendre ce routeur.
    p_prepare = subparsers.add_parser("prepare", help="Préparer audio et manifests")
    add_common_args(p_prepare)
    p_prepare.add_argument("--langpair", required=True)
    p_prepare.add_argument(
        "--input-root", type=Path, default=PROJECT_ROOT / "datasets" / "raw"
    )
    p_prepare.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "datasets" / "processed"
    )
    p_prepare.add_argument(
        "--manifests-root", type=Path, default=PROJECT_ROOT / "datasets" / "manifests"
    )
    p_prepare.add_argument("--sample-rate", type=int, default=16000)
    p_prepare.add_argument("--min-duration", type=float, default=1.0)
    p_prepare.add_argument("--max-duration", type=float, default=30.0)
    p_prepare.add_argument("--text-norm", default="nfkc", choices=("nfkc", "none"))
    p_prepare.add_argument("--lowercase", action="store_true", default=False)
    p_prepare.add_argument(
        "--segment-mode",
        choices=("utterance", "sentence_like"),
        default="utterance",
    )
    p_prepare.add_argument("--sentence-target-duration", type=float, default=10.0)
    p_prepare.add_argument("--sentence-max-duration", type=float, default=15.0)
    p_prepare.add_argument(
        "--sentence-require-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_prepare.add_argument(
        "--fail-on-leak", action=argparse.BooleanOptionalAction, default=True
    )
    p_prepare.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    p_prepare.add_argument("--report", type=Path, default=None)
    p_prepare.add_argument("--verify-only", action="store_true")
    p_prepare.set_defaults(func=cmd_prepare)

    p_run = subparsers.add_parser("run", help="Enchaîner preflight → prepare")
    add_common_args(p_run)
    p_run.add_argument("--langpair", default="fr-en")
    p_run.add_argument("--from-stage", default="preflight", choices=STAGES)
    p_run.add_argument("--to-stage", default="prepare", choices=STAGES)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
