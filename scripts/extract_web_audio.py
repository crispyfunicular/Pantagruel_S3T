#!/usr/bin/env python3
"""Extraire les segments audio du carrousel d'exemples vers ``docs/audio/``.

Copie les cinq WAV utterance référencés par ``docs/index.html`` depuis le corpus
préparé (``datasets/processed/<langpair>/<split>/``) vers ``docs/audio/``.

Prérequis : étape ``2_prepare`` exécutée en mode ``utterance`` pour la paire cible.

Usage :
    python scripts/extract_web_audio.py
    python scripts/extract_web_audio.py --langpair fr-en --processed-root datasets/processed
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Identifiants alignés sur docs/index.html (PHRASES) et documentation/phrases.md.
WEB_EXAMPLE_IDS: tuple[str, ...] = (
    "9fxo9YJhnG8_5",
    "9fxo9YJhnG8_6",
    "9fxo9YJhnG8_7",
    "9fxo9YJhnG8_8",
    "9fxo9YJhnG8_9",
)

SPLITS: tuple[str, ...] = ("train", "valid", "test")


def find_processed_wav(
    processed_root: Path,
    langpair: str,
    utt_id: str,
) -> Path | None:
    """Localiser un WAV segmenté dans train, valid ou test.

    Args:
        processed_root: Racine ``datasets/processed`` (ou équivalent).
        langpair: Paire linguistique (ex. ``fr-en``).
        utt_id: Identifiant utterance m-TEDx.

    Returns:
        Chemin du fichier s'il existe, sinon ``None``.
    """
    for split in SPLITS:
        candidate = processed_root / langpair / split / f"{utt_id}.wav"
        if candidate.is_file():
            return candidate
    return None


def extract_web_audio(
    *,
    processed_root: Path,
    web_audio_dir: Path,
    langpair: str,
    utt_ids: tuple[str, ...] = WEB_EXAMPLE_IDS,
) -> list[tuple[str, Path]]:
    """Copier les segments vers ``docs/audio/``.

    Args:
        processed_root: Racine des WAV préparés.
        web_audio_dir: Dossier de sortie (créé si besoin).
        langpair: Paire linguistique source.
        utt_ids: Liste d'identifiants à exporter.

    Returns:
        Paires ``(utt_id, chemin_source)`` pour chaque copie réussie.

    Raises:
        FileNotFoundError: Si un identifiant est introuvable dans le corpus préparé.
    """
    web_audio_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, Path]] = []

    for utt_id in utt_ids:
        source = find_processed_wav(processed_root, langpair, utt_id)
        if source is None:
            raise FileNotFoundError(
                f"Segment introuvable : {utt_id}.wav sous "
                f"{processed_root / langpair}/{{train,valid,test}}/ — "
                "lancer ``python scripts_communs/2_prepare.py`` (utterance) d'abord."
            )
        dest = web_audio_dir / f"{utt_id}.wav"
        shutil.copy2(source, dest)
        copied.append((utt_id, source))

    return copied


def build_parser() -> argparse.ArgumentParser:
    """Construire l'analyseur CLI."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Copier les WAV d'exemples m-TEDx vers docs/audio/."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=root / "datasets" / "processed",
        help="Racine des WAV utterance (défaut : datasets/processed).",
    )
    parser.add_argument(
        "--web-audio-dir",
        type=Path,
        default=root / "docs" / "audio",
        help="Dossier de sortie du site (défaut : docs/audio).",
    )
    parser.add_argument(
        "--langpair",
        default="fr-en",
        help="Paire linguistique m-TEDx (défaut : fr-en).",
    )
    return parser


def main() -> int:
    """Point d'entrée CLI."""
    args = build_parser().parse_args()
    try:
        copied = extract_web_audio(
            processed_root=args.processed_root,
            web_audio_dir=args.web_audio_dir,
            langpair=args.langpair,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    for utt_id, source in copied:
        print(f"  {utt_id}.wav  ←  {source}")
    print(f"OK — {len(copied)} fichier(s) dans {args.web_audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
