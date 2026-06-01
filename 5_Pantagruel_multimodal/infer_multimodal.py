#!/usr/bin/env python3
"""
Inférence variante 5 — checkpoint entraîné avec encodeur Speech_Text.

Délègue à ``1_Transformer/6_infer.py`` (décodage glouton + JSONL).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_MULTIMODAL_ROOT = Path(__file__).resolve().parent
if str(_MULTIMODAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIMODAL_ROOT))

from multimodal_common import delegate_transformer_infer  # noqa: E402


def run_infer_multimodal(
    *,
    checkpoint: Path,
    input_audio: Path,
    config_path: Path | None,
    beam_size: int,
    output: Path,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """Inférer une traduction anglaise depuis un WAV français."""
    if dry_run:
        print("[dry-run] infer_multimodal -> 1_Transformer/6_infer.py")
        print(f"  checkpoint:{checkpoint}")
        print(f"  input:     {input_audio}")
        print(f"  config:    {config_path}")
        print(f"  beam:      {beam_size}")
        print(f"  output:    {output}")
        print(f"  prefer_cpu:{prefer_cpu}")
    return delegate_transformer_infer(
        checkpoint=checkpoint,
        input_audio=input_audio,
        config_path=config_path,
        beam_size=beam_size,
        output=output,
        dry_run=dry_run,
        verbose=verbose,
        prefer_cpu=prefer_cpu,
    )


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape `infer_multimodal`."""
    parser = argparse.ArgumentParser(
        description="Pantagruel multimodal — inférence WAV"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "inference"
        / "predictions.jsonl",
    )
    parser.add_argument("--prefer-cpu", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée partagé (CLI directe ou routeur de pipeline)."""
    return run_infer_multimodal(
        checkpoint=args.checkpoint,
        input_audio=args.input_audio,
        config_path=args.config,
        beam_size=args.beam_size,
        output=args.output,
        dry_run=args.dry_run,
        verbose=args.verbose,
        prefer_cpu=args.prefer_cpu,
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI autonome."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
