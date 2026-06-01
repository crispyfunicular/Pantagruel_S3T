#!/usr/bin/env python3
"""
Évaluation variante 5 — Pantagruel multimodal + SacreBLEU (dev/test).

Délègue à ``1_Transformer/5_evaluate.py`` puis met à jour ``experiments_tracking.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MULTIMODAL_ROOT = Path(__file__).resolve().parent
if str(_MULTIMODAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIMODAL_ROOT))

from multimodal_common import delegate_transformer_evaluate  # noqa: E402


def run_evaluate_multimodal(
    *,
    config_path: Path,
    run_id: str,
    checkpoint: Path | None,
    output_dir: Path | None,
    beam_size: int,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """Lancer l'évaluation SacreBLEU (délégation Transformer)."""
    if dry_run:
        print("[dry-run] evaluate_multimodal -> 1_Transformer/5_evaluate.py")
        print(f"  config:    {config_path}")
        print(f"  run_id:    {run_id}")
        print(f"  checkpoint:{checkpoint}")
        print(f"  output:    {output_dir}")
        print(f"  beam:      {beam_size}")
        print(f"  prefer_cpu:{prefer_cpu}")
    return delegate_transformer_evaluate(
        config_path=config_path,
        run_id=run_id,
        checkpoint=checkpoint,
        output_dir=output_dir,
        beam_size=beam_size,
        dry_run=dry_run,
        verbose=verbose,
        prefer_cpu=prefer_cpu,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape `evaluate_multimodal`."""
    parser = argparse.ArgumentParser(
        description="Pantagruel multimodal — évaluation SacreBLEU"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--prefer-cpu", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée partagé (CLI directe ou routeur de pipeline)."""
    return run_evaluate_multimodal(
        config_path=args.config,
        run_id=args.run_id,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        beam_size=args.beam_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
        prefer_cpu=args.prefer_cpu,
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI autonome."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
