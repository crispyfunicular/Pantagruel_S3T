#!/usr/bin/env python3
"""
Construire le manifest TSV pour le corpus oralité pluriTAL (audio français externe).

Entrée par défaut : ~/git/perso/pluriTAL/oralite/projet_final/corpus/
Sortie : datasets/external/oralite_fr/manifest.tsv

Usage :
  python scripts/build_oralite_manifest.py
  python scripts/build_oralite_manifest.py --corpus-dir /chemin/vers/corpus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_communs.external_corpus import (  # noqa: E402
    discover_wav_lab_pairs,
    write_external_manifest,
)

DEFAULT_CORPUS = Path.home() / "git/perso/pluriTAL/oralite/projet_final/corpus"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets/external/oralite_fr/manifest.tsv"


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Manifest TSV pour corpus oralité (WAV 16 kHz + .lab FR)",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Répertoire des .wav/.lab (défaut: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Manifest de sortie (défaut: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    items = discover_wav_lab_pairs(args.corpus_dir)
    if not items:
        print(f"ERROR: aucun .wav dans {args.corpus_dir}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(f"[dry-run] {len(items)} fichiers WAV")
        for item in items[:5]:
            print(
                f"  {item.sample_id}: {item.audio_path.name} | FR: {item.src_text[:60]}…"
            )
        if len(items) > 5:
            print(f"  … ({len(items) - 5} de plus)")
        return 0

    out = write_external_manifest(items, args.output)
    print(f"Manifest écrit: {out} ({len(items)} clips)")
    print("Suite: python scripts/run_external_corpus_infer.py --manifest", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
