#!/usr/bin/env python3
"""
Entraînement variante 5 — Pantagruel multimodal (`speech_text`) + décodeur Transformer.

Délègue à ``1_Transformer/4_train.py`` avec encodeur HF ``Speech_Text_*`` (API ``mode=AUDIO``)
et manifests ``sentence_like``. Voir ``configs/fr-en/base.yaml``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MULTIMODAL_ROOT = Path(__file__).resolve().parent
if str(_MULTIMODAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIMODAL_ROOT))

from multimodal_common import delegate_transformer_train  # noqa: E402


def run_train_multimodal(
    *,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """Lancer l'entraînement ST (délégation Transformer)."""
    if dry_run:
        print("[dry-run] train_multimodal -> 1_Transformer/4_train.py")
        print(f"  config:    {config_path}")
        print(f"  run_id:    {run_id}")
        print(f"  output:    {output_dir}")
        print(f"  prefer_cpu:{prefer_cpu}")
    return delegate_transformer_train(
        config_path=config_path,
        run_id=run_id,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        prefer_cpu=prefer_cpu,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape `train_multimodal`."""
    parser = argparse.ArgumentParser(
        description="Pantagruel multimodal — entraînement ST (encodeur Speech_Text + décodeur)"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefer-cpu", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée partagé (CLI directe ou routeur de pipeline)."""
    return run_train_multimodal(
        config_path=args.config,
        run_id=args.run_id,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        prefer_cpu=args.prefer_cpu,
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI autonome."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
