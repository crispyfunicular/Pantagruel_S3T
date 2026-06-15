#!/usr/bin/env python3
"""
Étape 6 — Inférence sur des fichiers WAV français arbitraires (chemin production).

Contrairement à l'étape 5, ne requiert pas les splits m-TEDx : charge un checkpoint
fine-tuné, encode un WAV 16 kHz fourni par l'utilisateur, décode le texte anglais
(greedy ou beam search selon la config) et ajoute un enregistrement JSONL pour audit.

Entrées :
    - ``--checkpoint`` (``best.pt`` de l'étape 4).
    - Chemin WAV ``--input-audio``.
    - ``--config`` optionnel si le checkpoint n'a pas de dict config embarqué.
    - Chemin modèle SPM depuis ``data.spm_model`` de la config.

Sorties :
    - Lignes JSONL ajoutées à ``--output`` (défaut ``inference/predictions.jsonl``).

Codes de sortie : 0 succès, 2 checkpoint/audio/config/SPM manquant.
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
from scripts_communs.st_common import (
    PROJECT_ROOT,
    build_s3t_model,
    decode_batch,
    decode_ids_to_text,
    deep_get,
    load_sentencepiece,
    load_waveform,
    load_yaml_config,
)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Charger le checkpoint étape 4 ; même contrat que ``5_evaluate.load_checkpoint``."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def run_infer(
    *,
    checkpoint: Path,
    input_audio: Path,
    config_path: Path | None,
    beam_size: int | None,
    output: Path,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """
    Traduire un fichier audio et ajouter le résultat à un journal JSONL.

    Paramètres :
        checkpoint : ``best.pt`` fine-tuné.
        input_audio : WAV parole française (16 kHz mono attendu).
        config_path : YAML de repli si le checkpoint n'a pas de clé ``config``.
        beam_size : Largeur du faisceau (défaut : decode.beam_size du YAML).
        output : Chemin JSONL à compléter.
        dry_run, verbose, prefer_cpu : Drapeaux CLI.

    Retour :
        0 on success, 2 on missing inputs.
    """
    start_wall_s = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()
    payload = load_checkpoint(checkpoint)
    config = payload.get("config")
    if not isinstance(config, dict):
        if config_path is None:
            print(
                "ERROR: checkpoint has no embedded config and --config was not provided",
                file=sys.stderr,
            )
            return 2
        config = load_yaml_config(config_path)

    spm_model_path = PROJECT_ROOT / str(deep_get(config, "data.spm_model"))
    if not spm_model_path.is_file():
        print(f"ERROR: missing SentencePiece model: {spm_model_path}", file=sys.stderr)
        return 2
    if not input_audio.is_file():
        print(f"ERROR: missing input audio: {input_audio}", file=sys.stderr)
        return 2

    sample_rate = int(deep_get(config, "data.sample_rate", 16000))
    max_target_tokens = int(deep_get(config, "train.max_target_tokens", 256))
    max_new_tokens = int(deep_get(config, "decode.max_len_b", 128))
    effective_beam_size = int(
        deep_get(config, "decode.beam_size", 5) if beam_size is None else beam_size
    )

    if dry_run:
        print("[dry-run] infer stage plan:")
        print(f"  checkpoint: {checkpoint}")
        print(f"  input:      {input_audio}")
        print(f"  spm_model:  {spm_model_path}")
        print(f"  output:     {output}")
        print(f"  beam_size:  {effective_beam_size}")
        return 0

    sp_model = load_sentencepiece(spm_model_path)
    pad_id = int(payload.get("pad_id", sp_model.pad_id()))
    bos_id = int(payload.get("bos_id", sp_model.bos_id()))
    eos_id = int(payload.get("eos_id", sp_model.eos_id()))
    vocab_size = int(payload.get("vocab_size", sp_model.get_piece_size()))

    device = torch.device(
        "cpu" if prefer_cpu or not torch.cuda.is_available() else "cuda"
    )
    model = build_s3t_model(
        config,
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_positions=max_target_tokens + 2,
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=False)
    model.eval()

    wave = load_waveform(input_audio, sample_rate).unsqueeze(0).to(device)
    attn = torch.ones((1, wave.size(1)), dtype=torch.long, device=device)
    generated = decode_batch(
        model=model,
        input_values=wave,
        attention_mask=attn,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
        beam_size=effective_beam_size,
    )
    prediction = decode_ids_to_text(
        generated[0].tolist(),
        sp_model=sp_model,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "start_timestamp_utc": start_utc,
        "end_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "duration_s": float(time.time() - start_wall_s),
        "device": str(device),
        "input_audio": str(input_audio.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "prediction": prediction,
        "beam_size": effective_beam_size,
    }
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if verbose:
        print(f"Prediction: {prediction}")
    print(f"Inference complete. Output appended to {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI pour l'étape 6."""
    parser = argparse.ArgumentParser(
        description="S3T Étape 6 — Inférence sur nouvel audio",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-audio", type=Path, required=True)
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Largeur du faisceau (défaut : decode.beam_size du YAML, sinon 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    parser.add_argument("--prefer-cpu", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par ``pipeline.py infer``."""
    checkpoint = getattr(args, "checkpoint", None)
    input_audio = getattr(args, "input_audio", None)
    if checkpoint is None or input_audio is None:
        print(
            "ERROR: --checkpoint and --input-audio are required for infer stage",
            file=sys.stderr,
        )
        return 2
    return run_infer(
        checkpoint=checkpoint,
        input_audio=input_audio,
        config_path=getattr(args, "config", None),
        beam_size=getattr(args, "beam_size", None),
        output=getattr(
            args, "output", PROJECT_ROOT / "inference" / "predictions.jsonl"
        ),
        dry_run=getattr(args, "dry_run", False),
        verbose=getattr(args, "verbose", False),
        prefer_cpu=getattr(args, "prefer_cpu", False),
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
