#!/usr/bin/env python3
"""
Inférence speechLLM — traduire un WAV français arbitraire via checkpoint B1.

Sortie : ligne JSONL append dans ``--output`` (défaut ``inference/predictions.jsonl``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from speechLLM.speechllm_lib import (
    PROJECT_ROOT,
    deep_get,
    load_projector_checkpoint,
    load_speechllm_checkpoint,
    load_speechllm_from_config,
    load_waveform,
    load_yaml_config,
    resolve_speechllm_config_path,
)


def run_infer(
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
    """Traduire un fichier audio et journaliser le résultat."""
    if dry_run:
        print("[dry-run] speechLLM infer:")
        print(f"  checkpoint:   {checkpoint}")
        print(f"  input_audio:  {input_audio}")
        print(f"  output:       {output}")
        return 0

    if not checkpoint.is_file():
        print(f"ERROR: missing checkpoint: {checkpoint}", file=sys.stderr)
        return 2
    if not input_audio.is_file():
        print(f"ERROR: missing input audio: {input_audio}", file=sys.stderr)
        return 2

    start_wall_s = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()

    payload = load_speechllm_checkpoint(checkpoint)
    config = payload.get("config")
    if not isinstance(config, dict):
        if config_path is None:
            print(
                "ERROR: checkpoint has no embedded config and --config was not provided",
                file=sys.stderr,
            )
            return 2
        config = load_yaml_config(resolve_speechllm_config_path(config_path))

    sample_rate = int(deep_get(config, "data.sample_rate", 16000))
    prompt = str(
        deep_get(
            config,
            "prompt.template",
            "Translate the French speech to English.",
        )
    )
    max_new_tokens = int(deep_get(config, "decode.max_new_tokens", 128))
    num_beams = (
        beam_size if beam_size > 0 else int(deep_get(config, "decode.beam_size", 4))
    )

    device = torch.device(
        "cpu" if prefer_cpu or not torch.cuda.is_available() else "cuda"
    )
    model = load_speechllm_from_config(config, device=device)
    load_projector_checkpoint(model, payload)
    model.eval()

    wave = load_waveform(input_audio, sample_rate)
    input_values = wave.unsqueeze(0).to(device)
    attention_mask = torch.ones((1, wave.numel()), dtype=torch.long, device=device)

    with torch.no_grad():
        hypotheses = model.generate_text_batch(
            input_values,
            attention_mask,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
    translation = hypotheses[0] if hypotheses else ""

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "start_timestamp_utc": start_utc,
        "end_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": float(time.time() - start_wall_s),
        "pipeline": "speechllm",
        "input_audio": str(input_audio.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "translation": translation,
        "beam_size": num_beams,
        "device": str(device),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if verbose:
        print(json.dumps(record, ensure_ascii=False, indent=2))
    else:
        print(translation)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape `infer` speechLLM."""
    parser = argparse.ArgumentParser(description="speechLLM — inférence WAV")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--beam-size", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    parser.add_argument("--prefer-cpu", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par `2_speechLLM/pipeline.py infer`."""
    return run_infer(
        checkpoint=args.checkpoint,
        input_audio=args.input_audio,
        config_path=getattr(args, "config", None),
        beam_size=getattr(args, "beam_size", 0),
        output=args.output,
        dry_run=args.dry_run,
        verbose=args.verbose,
        prefer_cpu=args.prefer_cpu,
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
