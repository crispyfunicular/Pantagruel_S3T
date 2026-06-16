#!/usr/bin/env python3
"""
Inférence cascade — traduire un WAV français arbitraire via ASR puis MT.

Sortie : ligne JSONL append dans ``--output`` (défaut ``inference/predictions.jsonl``),
en cohérence avec ``2_speechLLM/infer.py`` et ``3_Gemini/infer_gemini.py``.

État : squelette — ``--dry-run`` opérationnel ; l'inférence réelle attend les backends.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Cascade.cascade_common import (
    EXIT_CONFIG,
    EXIT_NOT_IMPLEMENTED,
    EXIT_SUCCESS,
    CascadePipelineNotReadyError,
    cascade_translate_audio,
    load_cascade_settings,
    resolve_cascade_config_path,
)
from speechLLM.speechllm_lib import load_yaml_config


def run_infer_cascade(
    *,
    input_audio: Path,
    config_path: Path | None,
    output: Path,
    dry_run: bool,
    verbose: bool,
) -> int:
    """
    Traduire un fichier audio via ASR→MT et journaliser le résultat JSONL.

    Paramètres :
        input_audio : WAV d'entrée (16 kHz mono recommandé).
        config_path : YAML optionnel (modèles ASR/MT).
        output : Fichier JSONL de sortie (append).
        dry_run : Afficher le plan sans inférence.
        verbose : Logs détaillés.

    Retour :
        Code de sortie (0, 2 ou 3).
    """
    if dry_run:
        print("[dry-run] cascade infer:")
        print(f"  input_audio: {input_audio}")
        print(f"  output:      {output}")
        print(f"  config:      {config_path or '(defaults)'}")
        return EXIT_SUCCESS

    if not input_audio.is_file():
        print(f"ERROR: missing input audio: {input_audio}", file=sys.stderr)
        return EXIT_CONFIG

    config: dict[str, Any] = {}
    settings = load_cascade_settings(config)
    if config_path is not None:
        config = load_yaml_config(resolve_cascade_config_path(config_path))
        settings = load_cascade_settings(config)

    if verbose:
        print(f"Infer cascade: ASR={settings.asr_model_id}, MT={settings.mt_model_id}")

    try:
        hypothesis = cascade_translate_audio(input_audio, settings)
    except CascadePipelineNotReadyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "cascade_asr_mt",
        "input_audio": str(input_audio.resolve()),
        "hypothesis": hypothesis,
        "asr_model_id": settings.asr_model_id,
        "mt_model_id": settings.mt_model_id,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if verbose:
        print(f"Wrote: {output}")
    print(hypothesis)
    return EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape ``infer-cascade``."""
    from speechLLM.speechllm_lib import PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Cascade ASR→MT — inférence WAV")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée partagé (CLI directe ou routeur)."""
    return run_infer_cascade(
        input_audio=args.input_audio,
        config_path=args.config,
        output=args.output,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    """``main`` CLI autonome."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
