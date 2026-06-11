#!/usr/bin/env python3
"""
Construire le manifest TSV du corpus personnel (lectures FR, références EN).

Entrée :
    - ``corpus_perso/*.wav`` (``N-K.wav``, textes distincts)
    - ``corpus_perso/corpus_perso_ref_EN.txt``

Sortie :
    - ``corpus_perso/corpus_perso_test.tsv``

Usage :
    python scripts/build_corpus_perso_manifest.py
    python scripts/build_corpus_perso_manifest.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_communs.corpus_perso import (  # noqa: E402
    build_corpus_perso_items,
    write_corpus_perso_manifest,
)

DEFAULT_CORPUS_DIR = PROJECT_ROOT / "corpus_perso"
DEFAULT_REFERENCE = DEFAULT_CORPUS_DIR / "corpus_perso_ref_EN.txt"
DEFAULT_OUTPUT = DEFAULT_CORPUS_DIR / "corpus_perso_test.tsv"


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Manifest TSV pour corpus perso (WAV FR + ref EN)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help=f"Dossier WAV + référence (défaut: {DEFAULT_CORPUS_DIR})",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Fichier de références anglaises",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Manifest de sortie (défaut: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.reference.is_file():
        print(f"ERROR: référence absente: {args.reference}", file=sys.stderr)
        return 2

    try:
        items = build_corpus_perso_items(
            corpus_dir=args.corpus_dir,
            reference_path=args.reference,
            project_root=PROJECT_ROOT,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[dry-run] {len(items)} clips → {args.output}")
        for item in items[:3]:
            print(
                f"  {item.sample_id} → para {item.paragraph_index + 1}: "
                f"{item.tgt_text[:72]}…"
            )
        if len(items) > 3:
            print(f"  … ({len(items) - 3} de plus)")
        return 0

    out = write_corpus_perso_manifest(items, args.output)
    print(f"Manifest écrit: {out} ({len(items)} clips)")
    print("Suite: python scripts/run_corpus_perso_eval.py --dry-run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
