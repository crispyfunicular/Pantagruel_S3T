#!/usr/bin/env python3
"""
CLI pipeline ST unifié pour la réplication de traduction vocale Pantagruel.

Ce fichier est un **routeur uniquement** : il analyse les sous-commandes et délègue aux modules
numérotés sous ``scripts/`` via import dynamique. Aucune logique d'entraînement ou de données ici.

Étapes :
    preflight   — Valider machine, GPU, disque, réseau
    download    — Récupérer corpus m-TEDx (OpenSLR-100)
    prepare     — Audio WAV 16 kHz, manifests, normalisation texte
    spm         — Entraîner tokenizers SentencePiece (split train uniquement)
    train       — Entraîner modèle ST (encodeur + décodeur)
    evaluate    — Décoder valid/test + métriques SacreBLEU
    infer       — Inférence sur nouveaux fichiers audio
    run         — Enchaîner les étapes bout en bout

Usage :
    python scripts/pipeline.py preflight
    python scripts/pipeline.py download --langpairs fr-es
    python scripts/pipeline.py prepare --langpair fr-es
    python scripts/pipeline.py spm --langpair fr-es --vocab-size 1000
    python scripts/pipeline.py train --config configs/fr-es/base.yaml --run-id run_001
    python scripts/pipeline.py evaluate --config configs/fr-es/base.yaml --run-id run_001
    python scripts/pipeline.py infer --checkpoint runs/fr-es/run_001/checkpoints/best.pt --input-audio audio.wav
    python scripts/pipeline.py run --langpair fr-es --run-id run_001 --from-stage preflight --to-stage evaluate
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGES = ("preflight", "download", "prepare", "spm", "train", "evaluate", "infer")

EXIT_NOT_IMPLEMENTED = 7


class NotYetImplemented(NotImplementedError):
    """Levée lorsqu'un gestionnaire d'étape pipeline est encore un placeholder."""


def not_yet(stage: str) -> None:
    """Ancien helper pour étapes non implémentées (préférer les vrais modules d'étape)."""
    raise NotYetImplemented(
        f"NotYetImplemented: stage '{stage}' is not implemented yet."
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attacher ``--verbose``, ``--dry-run`` et ``--log-file`` à un sous-parseur."""
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Journalisation verbeuse.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher les actions prévues sans exécuter.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Chemin optionnel du fichier journal.",
    )


def _load_preflight_module():
    """Importer l'étape 0 sans installation package (les scripts ne sont pas toujours un package)."""
    path = PROJECT_ROOT / "scripts" / "0_preflight.py"
    spec = importlib.util.spec_from_file_location("s3t_preflight", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preflight module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_preflight(args: argparse.Namespace) -> int:
    """Exécuter les vérifications préalables étape 0."""
    preflight = _load_preflight_module()
    return preflight.run_from_namespace(args)


def _load_download_module():
    """Importer le module téléchargement étape 1."""
    path = PROJECT_ROOT / "scripts" / "1_download.py"
    spec = importlib.util.spec_from_file_location("s3t_download", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load download module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_download(args: argparse.Namespace) -> int:
    """Exécuter le téléchargement m-TEDx étape 1."""
    download = _load_download_module()
    return download.run_from_namespace(args)


def _load_prepare_module():
    """Importer le module prepare étape 2."""
    path = PROJECT_ROOT / "scripts" / "2_prepare.py"
    spec = importlib.util.spec_from_file_location("s3t_prepare", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prepare module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_prepare(args: argparse.Namespace) -> int:
    """Exécuter la préparation audio/manifests étape 2."""
    prepare = _load_prepare_module()
    return prepare.run_from_namespace(args)


def _load_spm_module():
    """Importer le module SentencePiece étape 3."""
    path = PROJECT_ROOT / "scripts" / "3_spm.py"
    spec = importlib.util.spec_from_file_location("s3t_spm", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spm module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_train_module():
    """Importer le module train étape 4."""
    path = PROJECT_ROOT / "scripts" / "4_train.py"
    spec = importlib.util.spec_from_file_location("s3t_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load train module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_evaluate_module():
    """Importer le module evaluate étape 5."""
    path = PROJECT_ROOT / "scripts" / "5_evaluate.py"
    spec = importlib.util.spec_from_file_location("s3t_evaluate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluate module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_infer_module():
    """Importer le module infer étape 6."""
    path = PROJECT_ROOT / "scripts" / "6_infer.py"
    spec = importlib.util.spec_from_file_location("s3t_infer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load infer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_spm(args: argparse.Namespace) -> int:
    """Exécuter l'entraînement SPM étape 3."""
    spm_stage = _load_spm_module()
    return spm_stage.run_from_namespace(args)


def cmd_train(args: argparse.Namespace) -> int:
    """Exécuter l'entraînement ST étape 4."""
    train_stage = _load_train_module()
    return train_stage.run_from_namespace(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Exécuter l'évaluation SacreBLEU étape 5."""
    evaluate_stage = _load_evaluate_module()
    return evaluate_stage.run_from_namespace(args)


def cmd_infer(args: argparse.Namespace) -> int:
    """Exécuter l'inférence étape 6 sur nouvel audio."""
    infer_stage = _load_infer_module()
    return infer_stage.run_from_namespace(args)


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
    """Orchestrer les étapes pipeline de --from-stage à --to-stage."""
    from_idx = STAGES.index(args.from_stage)
    to_idx = STAGES.index(args.to_stage)
    if from_idx > to_idx:
        print(
            f"ERROR: --from-stage ({args.from_stage}) must come before --to-stage ({args.to_stage})",
            file=sys.stderr,
        )
        return 2

    stage_handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "preflight": cmd_preflight,
        "download": cmd_download,
        "prepare": cmd_prepare,
        "spm": cmd_spm,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
    }

    selected = STAGES[from_idx : to_idx + 1]
    print("=" * 60)
    print("S3T PIPELINE RUN")
    print(f"  langpair: {args.langpair}")
    print(f"  run-id:   {args.run_id}")
    print(f"  stages:   {' → '.join(selected)}")
    print("=" * 60)

    for stage in selected:
        try:
            code = _run_stage(stage, stage_handlers[stage], args)
            if code != 0:
                return code
        except NotYetImplemented as exc:
            print(f"  {exc}", file=sys.stderr)
            return EXIT_NOT_IMPLEMENTED

    print("\n" + "=" * 60)
    print("PIPELINE RUN COMPLETE")
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    Construire les sous-parseurs et dispatcher vers le handler d'étape sélectionné.

    Retour :
        Code de sortie de l'étape, ou 7 si ``NotYetImplemented`` est levée.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline S3T — Traduction vocale Pantagruel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- preflight ---
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
    p_preflight.add_argument(
        "--disk-path",
        type=Path,
        default=PROJECT_ROOT,
    )
    p_preflight.add_argument("--network-timeout", type=float, default=10.0)
    p_preflight.set_defaults(func=cmd_preflight)

    # --- download ---
    p_download = subparsers.add_parser(
        "download", help="Télécharger jeux m-TEDx (OpenSLR-100)"
    )
    add_common_args(p_download)
    p_download.add_argument(
        "--langpairs",
        default="fr-en",
        help="Paires de langues séparées par virgules (défaut : fr-en)",
    )
    p_download.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw",
    )
    p_download.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_download.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_download.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "download_manifest.json",
    )
    p_download.set_defaults(func=cmd_download)

    # --- prepare ---
    p_prepare = subparsers.add_parser("prepare", help="Préparer audio et manifests")
    add_common_args(p_prepare)
    p_prepare.add_argument("--langpair", required=True, help="e.g. fr-es")
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
        "--fail-on-leak",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_prepare.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ignorer les segments dont le WAV de sortie valide existe déjà sur disque",
    )
    p_prepare.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Chemin rapport JSON (défaut : artifacts/prepare_<langpair>.json)",
    )
    p_prepare.add_argument(
        "--verify-only",
        action="store_true",
        help="Vérifier uniquement les WAV et manifests existants",
    )
    p_prepare.set_defaults(func=cmd_prepare)

    # --- spm ---
    p_spm = subparsers.add_parser("spm", help="Entraîner tokenizer SentencePiece")
    add_common_args(p_spm)
    p_spm.add_argument("--langpair", required=True)
    p_spm.add_argument("--vocab-size", type=int, default=1000)
    p_spm.add_argument("--model-type", default="unigram", choices=("unigram", "bpe"))
    p_spm.add_argument(
        "--manifests-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests",
    )
    p_spm.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "spm",
    )
    p_spm.add_argument("--train-text", type=Path, default=None)
    p_spm.add_argument("--character-coverage", type=float, default=1.0)
    p_spm.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p_spm.add_argument("--report", type=Path, default=None)
    p_spm.set_defaults(func=cmd_spm)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Entraîner modèle ST")
    add_common_args(p_train)
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--run-id", required=True)
    p_train.add_argument("--output-dir", type=Path, default=None)
    p_train.add_argument("--prefer-cpu", action="store_true", default=False)
    p_train.set_defaults(func=cmd_train)

    # --- evaluate ---
    p_evaluate = subparsers.add_parser("evaluate", help="Évaluer avec SacreBLEU")
    add_common_args(p_evaluate)
    p_evaluate.add_argument("--config", type=Path, required=True)
    p_evaluate.add_argument("--run-id", required=True)
    p_evaluate.add_argument("--checkpoint", type=Path, default=None)
    p_evaluate.add_argument("--beam-size", type=int, default=5)
    p_evaluate.add_argument("--output-dir", type=Path, default=None)
    p_evaluate.add_argument("--prefer-cpu", action="store_true", default=False)
    p_evaluate.set_defaults(func=cmd_evaluate)

    # --- infer ---
    p_infer = subparsers.add_parser("infer", help="Inférence sur audio")
    add_common_args(p_infer)
    p_infer.add_argument("--config", type=Path, default=None)
    p_infer.add_argument("--checkpoint", type=Path, required=True)
    p_infer.add_argument("--input-audio", type=Path, required=True)
    p_infer.add_argument("--beam-size", type=int, default=5)
    p_infer.add_argument("--prefer-cpu", action="store_true", default=False)
    p_infer.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    p_infer.set_defaults(func=cmd_infer)

    # --- run ---
    p_run = subparsers.add_parser(
        "run", help="Exécuter les étapes pipeline bout en bout"
    )
    add_common_args(p_run)
    p_run.add_argument("--langpair", default="fr-es")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument(
        "--from-stage",
        default="preflight",
        choices=STAGES,
    )
    p_run.add_argument(
        "--to-stage",
        default="evaluate",
        choices=STAGES,
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except NotYetImplemented as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    sys.exit(main())
