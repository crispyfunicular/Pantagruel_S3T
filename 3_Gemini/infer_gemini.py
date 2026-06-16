#!/usr/bin/env python3
"""
Inférence Gemini — traduire un WAV français arbitraire via API.

Sortie : ligne JSONL append dans ``--output`` (défaut ``inference/predictions.jsonl``),
en cohérence avec `2_speechLLM/infer.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Gemini.gemini_common import (
    DEFAULT_GEMINI_MODEL_ID,
    DEFAULT_PROMPT,
    GeminiRequest,
    MissingGeminiApiKeyError,
    create_gemini_client,
    translate_audio,
)
from speechLLM.speechllm_lib import (
    PROJECT_ROOT,
    deep_get,
    load_yaml_config,
    resolve_speechllm_config_path,
)


def run_infer_gemini(
    *,
    input_audio: Path,
    config_path: Path | None,
    model_id: str | None,
    output: Path,
    max_retries: int,
    dry_run: bool,
    verbose: bool,
) -> int:
    """Traduire un fichier audio et journaliser le résultat."""
    if dry_run:
        print("[dry-run] gemini infer:")
        print(f"  input_audio:  {input_audio}")
        print(f"  output:       {output}")
        print(f"  model_id:     {model_id or '(from config/default)'}")
        return 0

    if not input_audio.is_file():
        print(f"ERROR: missing input audio: {input_audio}", file=sys.stderr)
        return 2

    config: dict[str, Any] = {}
    if config_path is not None:
        config = load_yaml_config(resolve_speechllm_config_path(config_path))

    prompt = str(deep_get(config, "prompt.template", DEFAULT_PROMPT))
    effective_model_id = str(
        model_id or deep_get(config, "model.gemini_id", DEFAULT_GEMINI_MODEL_ID)
    )
    temperature = float(deep_get(config, "decode.temperature", 0.0))
    max_output_tokens = int(deep_get(config, "decode.max_output_tokens", 256))
    thinking_level_raw = deep_get(config, "decode.thinking_level", None)
    thinking_level = (
        str(thinking_level_raw).strip()
        if thinking_level_raw is not None and str(thinking_level_raw).strip()
        else None
    )

    try:
        client = create_gemini_client()
    except MissingGeminiApiKeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    request = GeminiRequest(
        model_id=effective_model_id,
        prompt=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_level=thinking_level,
    )

    last_error: str | None = None
    translation = ""
    for attempt in range(max(1, max_retries + 1)):
        try:
            translation = translate_audio(
                client=client,
                request=request,
                audio_path=input_audio,
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(
                    f"[gemini] retry {attempt}/{max_retries} for {input_audio}: {last_error}",
                    file=sys.stderr,
                )

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "gemini_st",
        "input_audio": str(input_audio.resolve()),
        "translation": translation,
        "model_id": effective_model_id,
        "prompt": prompt,
        "decode": {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "thinking_level": thinking_level,
            "max_retries": max_retries,
        },
    }
    if last_error is not None and not translation:
        record["error"] = last_error

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if verbose:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(translation)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape `infer-gemini`."""
    parser = argparse.ArgumentParser(description="Gemini ST — inférence WAV")
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par `2_speechLLM/pipeline.py infer-gemini`."""
    return run_infer_gemini(
        input_audio=args.input_audio,
        config_path=getattr(args, "config", None),
        model_id=getattr(args, "model_id", None),
        output=args.output,
        max_retries=int(getattr(args, "max_retries", 2)),
        dry_run=bool(getattr(args, "dry_run", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
