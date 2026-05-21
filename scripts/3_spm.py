#!/usr/bin/env python3
"""
Étape 3 — Entraîner un modèle de sous-mots SentencePiece sur les cibles anglaises uniquement.

Construit le vocabulaire côté cible pour la ST (traductions anglaises m-TEDx). Doit être
ajusté uniquement sur le texte du split d'entraînement pour éviter de fuir les statistiques
des splits dev/test (PRD).

Entrées :
    - ``datasets/manifests/<langpair>/train.target.txt`` (depuis l'étape 2).

Sorties :
    - ``datasets/processed/spm/<langpair>_<vocab>.model`` et ``.vocab``
    - Rapport JSON sous ``artifacts/spm_<langpair>_<vocab>.json``

Codes de sortie : 0 succès, 2 entrée manquante / mauvaise paire de langues,
3 sortie existante sans ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import sentencepiece as spm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_LANGPAIRS = frozenset({"fr-en", "fr-pt", "fr-es"})


def parse_langpair(value: str) -> str:
    """
    Valider la paire de langues par rapport aux directions m-TEDx supportées.

    Paramètres :
        value : ex. ``fr-en``.

    Retour :
        Chaîne de paire normalisée.

    Lève :
        ValueError : Paire inconnue.
    """
    pair = value.strip()
    if pair not in SUPPORTED_LANGPAIRS:
        supported = ", ".join(sorted(SUPPORTED_LANGPAIRS))
        raise ValueError(f"Unknown langpair: {pair}. Supported: {supported}")
    return pair


def run_spm(
    *,
    langpair: str,
    vocab_size: int,
    model_type: str,
    manifests_root: Path,
    output_dir: Path,
    train_text: Path | None,
    character_coverage: float,
    overwrite: bool,
    dry_run: bool,
    verbose: bool,
    report_path: Path | None,
) -> int:
    """
    Entraîner ou simuler (dry-run) SentencePiece sur le corpus cible d'entraînement.

    Paramètres :
        langpair : Nom du répertoire de la paire de langues.
        vocab_size : Taille du vocabulaire cible (ablations : 1k vs 5k).
        model_type : ``unigram`` ou ``bpe``.
        manifests_root : Racine contenant ``<langpair>/train.target.txt``.
        output_dir : Répertoire des sorties ``.model`` / ``.vocab``.
        train_text : Chemin optionnel vers le fichier texte d'entraînement.
        character_coverage : Hyperparamètre SPM de couverture des caractères.
        overwrite : Remplacer les fichiers modèle existants si True.
        dry_run, verbose : Drapeaux CLI.
        report_path : Chemin de sortie des métadonnées JSON.

    Retour :
        0 en cas de succès, 2 si entrée manquante, 3 si sorties existantes sans overwrite.
    """
    text_path = train_text or (manifests_root / langpair / "train.target.txt")
    if not text_path.is_file():
        print(f"ERROR: missing train text file: {text_path}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"{langpair}_{vocab_size}"
    model_path = prefix.with_suffix(".model")
    vocab_path = prefix.with_suffix(".vocab")
    report_path = report_path or (
        PROJECT_ROOT / "artifacts" / f"spm_{langpair}_{vocab_size}.json"
    )

    if (model_path.exists() or vocab_path.exists()) and not overwrite:
        print(
            "ERROR: SentencePiece output already exists. "
            "Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 3

    if dry_run:
        print("[dry-run] SentencePiece training plan:")
        print(f"  input:  {text_path}")
        print(f"  output: {model_path}")
        print(f"  vocab:  {vocab_path}")
        print(f"  type:   {model_type}")
        print(f"  size:   {vocab_size}")
        return 0

    if verbose:
        print("Training SentencePiece model...")
        print(f"  input={text_path}")
        print(f"  prefix={prefix}")

    spm.SentencePieceTrainer.train(
        input=text_path.as_posix(),
        model_prefix=prefix.as_posix(),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "langpair": langpair,
        "train_text": str(text_path.resolve()),
        "model_path": str(model_path.resolve()),
        "vocab_path": str(vocab_path.resolve()),
        "vocab_size": vocab_size,
        "model_type": model_type,
        "character_coverage": character_coverage,
        "exit_code": 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"SentencePiece complete. Model: {model_path}")
    print(f"Report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI pour l'étape 3."""
    parser = argparse.ArgumentParser(
        description="S3T Étape 3 — Entraînement du tokenizer SentencePiece",
    )
    parser.add_argument("--langpair", required=True, help="ex. fr-en")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--model-type", choices=("unigram", "bpe"), default="unigram")
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "spm",
    )
    parser.add_argument("--train-text", type=Path, default=None)
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par ``pipeline.py spm``."""
    try:
        langpair = parse_langpair(args.langpair)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return run_spm(
        langpair=langpair,
        vocab_size=getattr(args, "vocab_size", 1000),
        model_type=getattr(args, "model_type", "unigram"),
        manifests_root=getattr(
            args, "manifests_root", PROJECT_ROOT / "datasets" / "manifests"
        ),
        output_dir=getattr(
            args, "output_dir", PROJECT_ROOT / "datasets" / "processed" / "spm"
        ),
        train_text=getattr(args, "train_text", None),
        character_coverage=getattr(args, "character_coverage", 1.0),
        overwrite=getattr(args, "overwrite", False),
        dry_run=getattr(args, "dry_run", False),
        verbose=getattr(args, "verbose", False),
        report_path=getattr(args, "report", None),
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
