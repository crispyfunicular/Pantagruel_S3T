#!/usr/bin/env python3
"""
Inférence open baselines — traduire un WAV français arbitraire via un modèle ST open.

Sortie : ligne JSONL append dans ``--output`` (défaut ``inference/predictions.jsonl``),
en cohérence avec ``4_cascade/infer_cascade.py`` et ``3_Gemini/infer_gemini.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from OpenBaselines.open_common import (
    EXIT_CONFIG,
    EXIT_NOT_IMPLEMENTED,
    EXIT_SUCCESS,
    OpenPipelineNotReadyError,
    load_open_settings,
    open_translate_audio,
    resolve_open_config_path,
)
from speechLLM.speechllm_lib import load_yaml_config


def run_infer_open(
    *,
    input_audio: Path,
    config_path: Path | None,
    output: Path,
    dry_run: bool,
    verbose: bool,
) -> int:
    """
    Traduire un fichier audio via un modèle ST open et journaliser le résultat JSONL.

    Paramètres :
        input_audio : WAV d'entrée (16 kHz mono recommandé).
        config_path : YAML optionnel (modèle open).
        output : Fichier JSONL de sortie (append).
        dry_run : Afficher le plan sans inférence.
        verbose : Logs détaillés.

    Retour :
        Code de sortie (0, 2 ou 3).
    """
    if dry_run:
        print("[dry-run] open baselines infer:")
        print(f"  input_audio: {input_audio}")
        print(f"  output:      {output}")
        print(f"  config:      {config_path or '(defaults)'}")
        return EXIT_SUCCESS

    if not input_audio.is_file():
        print(f"ERROR: missing input audio: {input_audio}", file=sys.stderr)
        return EXIT_CONFIG

    config: dict[str, Any] = {}
    settings = load_open_settings(config)
    if config_path is not None:
        config = load_yaml_config(resolve_open_config_path(config_path))
        settings = load_open_settings(config)

    if verbose:
        print(
            f"Infer open baseline: type={settings.model_type}, "
            f"model={settings.model_id}"
        )

    try:
        hypothesis = open_translate_audio(input_audio, settings)
    except OpenPipelineNotReadyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "open_baselines_st",
        "input_audio": str(input_audio.resolve()),
        "hypothesis": hypothesis,
        "model_type": settings.model_type,
        "model_id": settings.model_id,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if verbose:
        print(f"Wrote: {output}")
    print(hypothesis)
    return EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape ``infer`` open baselines."""
    from speechLLM.speechllm_lib import PROJECT_ROOT

    parser = argparse.ArgumentParser(description="Open baselines ST — inférence WAV")
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
    return run_infer_open(
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
