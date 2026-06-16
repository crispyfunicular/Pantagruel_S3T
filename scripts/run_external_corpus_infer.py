#!/usr/bin/env python3
"""
Inférence ST fr→en sur un manifest externe (ex. corpus oralité pluriTAL).

Pas de SacreBLEU (pas de références anglaises) : sortie TSV ``id``, ``src_text`` (FR),
``en_hypothesis``, ``pipeline``.

Usage :
  python scripts/build_oralite_manifest.py
  python scripts/run_external_corpus_infer.py --variants speechllm,gemini
  python scripts/run_external_corpus_infer.py --variants cascade --limit 3
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_communs.st_common import (  # noqa: E402
    PROJECT_ROOT,
    deep_get,
    load_yaml_config,
    read_manifest,
)

DEFAULT_MANIFEST = PROJECT_ROOT / "datasets/external/oralite_fr/manifest.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets/external/oralite_fr/predictions"

SPEECHLLM_CONFIG = PROJECT_ROOT / "2_speechLLM/configs/fr-en/b1_sentence_long.yaml"
SPEECHLLM_CHECKPOINT = (
    PROJECT_ROOT
    / "runs/fr-en/run_005_speechllm_b1_sentence_long_unfreeze_encoder/checkpoints/best.pt"
)
GEMINI_CONFIG = PROJECT_ROOT / "3_Gemini/configs/fr-en/gemini_flash_sentence.yaml"
CASCADE_CONFIG = PROJECT_ROOT / "4_cascade/configs/fr-en/cascade_sentence.yaml"


def _write_results_tsv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Écrire les hypothèses EN + métadonnées FR."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "src_text_fr", "en_hypothesis", "pipeline", "audio"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _load_src_text_map(manifest_path: Path) -> dict[str, str]:
    """Lire ``src_text`` depuis le manifest (colonne optionnelle)."""
    mapping: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mapping[row["id"]] = (row.get("src_text") or "").strip()
    return mapping


def run_speechllm(
    samples: list,
    *,
    config_path: Path,
    checkpoint_path: Path,
    prefer_cpu: bool,
    verbose: bool,
) -> list[dict[str, str]]:
    """Traduire le manifest avec speechLLM B1."""
    from scripts_communs.variant_bootstrap import bootstrap_speechllm

    bootstrap_speechllm()
    from speechLLM.speechllm_lib import (
        collate_speechllm_batch,
        load_projector_checkpoint,
        load_speechllm_checkpoint,
        load_speechllm_from_config,
        resolve_speechllm_config_path,
    )
    from speechLLM.speechllm_lib import (
        deep_get as sl_deep_get,
    )

    config = load_yaml_config(resolve_speechllm_config_path(config_path))
    payload = load_speechllm_checkpoint(checkpoint_path)
    ckpt_config = payload.get("config", config)
    if not isinstance(ckpt_config, dict):
        ckpt_config = config

    device = torch.device(
        "cpu" if prefer_cpu or not torch.cuda.is_available() else "cuda"
    )
    model = load_speechllm_from_config(ckpt_config, device=device)
    load_projector_checkpoint(model, payload)
    model.eval()

    sample_rate = int(sl_deep_get(config, "data.sample_rate", 16000))
    prompt = str(
        sl_deep_get(
            config, "prompt.template", "Translate the French speech to English."
        )
    )
    max_new_tokens = int(sl_deep_get(config, "decode.max_new_tokens", 48))
    num_beams = int(sl_deep_get(config, "decode.beam_size", 1))

    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for sample in samples:
            batch = collate_speechllm_batch([sample], sample_rate=sample_rate)
            hyps = model.generate_text_batch(
                batch["input_values"].to(device),
                batch["attention_mask"].to(device),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            hyp = hyps[0] if hyps else ""
            if verbose:
                print(f"  {sample.sample_id}: {hyp[:80]}…")
            rows.append(
                {
                    "id": sample.sample_id,
                    "src_text_fr": "",
                    "en_hypothesis": hyp,
                    "pipeline": "speechllm",
                    "audio": str(sample.audio_path),
                }
            )
    return rows


def run_gemini(
    samples: list,
    *,
    config_path: Path,
    verbose: bool,
) -> list[dict[str, str]]:
    """Traduire le manifest via API Gemini."""
    from scripts_communs.variant_bootstrap import bootstrap_gemini

    bootstrap_gemini()
    from Gemini.gemini_common import (
        DEFAULT_GEMINI_MODEL_ID,
        DEFAULT_PROMPT,
        GeminiRequest,
        create_gemini_client,
        translate_audio_with_metadata,
    )
    from speechLLM.speechllm_lib import resolve_speechllm_config_path

    config = load_yaml_config(resolve_speechllm_config_path(config_path))
    model_id = str(deep_get(config, "model.gemini_id", DEFAULT_GEMINI_MODEL_ID))
    prompt = str(deep_get(config, "prompt.template", DEFAULT_PROMPT))
    temperature = float(deep_get(config, "decode.temperature", 0.0))
    max_output_tokens = int(deep_get(config, "decode.max_output_tokens", 256))

    client = create_gemini_client()
    request = GeminiRequest(
        model_id=model_id,
        prompt=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    rows: list[dict[str, str]] = []
    for sample in samples:
        result = translate_audio_with_metadata(
            client, sample.audio_path, request=request
        )
        hyp = (result.text or "").strip()
        if verbose:
            print(f"  {sample.sample_id}: {hyp[:80]}…")
        rows.append(
            {
                "id": sample.sample_id,
                "src_text_fr": "",
                "en_hypothesis": hyp,
                "pipeline": f"gemini:{model_id}",
                "audio": str(sample.audio_path),
            }
        )
    return rows


def run_cascade(
    samples: list,
    *,
    config_path: Path,
    verbose: bool,
) -> list[dict[str, str]]:
    """Traduire le manifest via cascade Whisper → Marian."""
    from scripts_communs.variant_bootstrap import bootstrap_cascade

    bootstrap_cascade()
    from Cascade.cascade_common import (
        cascade_translate_audio,
        load_cascade_settings,
        resolve_cascade_config_path,
    )

    config = load_yaml_config(resolve_cascade_config_path(config_path))
    settings = load_cascade_settings(config)

    rows: list[dict[str, str]] = []
    for sample in samples:
        hyp = cascade_translate_audio(sample.audio_path, settings).strip()
        if verbose:
            print(f"  {sample.sample_id}: {hyp[:80]}…")
        rows.append(
            {
                "id": sample.sample_id,
                "src_text_fr": "",
                "en_hypothesis": hyp,
                "pipeline": "cascade_asr_mt",
                "audio": str(sample.audio_path),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Inférence ST sur corpus externe (oralité, etc.)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variants",
        type=str,
        default="speechllm,gemini",
        help="Liste: speechllm, gemini, cascade",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max clips (0 = tous)")
    parser.add_argument("--speechllm-config", type=Path, default=SPEECHLLM_CONFIG)
    parser.add_argument(
        "--speechllm-checkpoint", type=Path, default=SPEECHLLM_CHECKPOINT
    )
    parser.add_argument("--gemini-config", type=Path, default=GEMINI_CONFIG)
    parser.add_argument("--cascade-config", type=Path, default=CASCADE_CONFIG)
    parser.add_argument("--prefer-cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(f"ERROR: manifest absent: {args.manifest}", file=sys.stderr)
        print("  Lancez: python scripts/build_oralite_manifest.py", file=sys.stderr)
        return 2

    samples = read_manifest(args.manifest)
    if args.limit > 0:
        samples = samples[: args.limit]
    src_map = _load_src_text_map(args.manifest)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.dry_run:
        print(f"[dry-run] {len(samples)} clips, variants={variants}")
        for sample in samples[:5]:
            fr = src_map.get(sample.sample_id, "")
            print(f"  {sample.sample_id}: {sample.audio_path.name} | {fr[:50]}")
        return 0

    for variant in variants:
        print(f"=== {variant} ({len(samples)} clips) ===")
        if variant == "speechllm":
            if not args.speechllm_checkpoint.is_file():
                print(
                    f"ERROR: checkpoint: {args.speechllm_checkpoint}", file=sys.stderr
                )
                return 2
            rows = run_speechllm(
                samples,
                config_path=args.speechllm_config,
                checkpoint_path=args.speechllm_checkpoint,
                prefer_cpu=args.prefer_cpu,
                verbose=args.verbose,
            )
            out_name = "speechllm.tsv"
        elif variant == "gemini":
            rows = run_gemini(
                samples, config_path=args.gemini_config, verbose=args.verbose
            )
            out_name = "gemini.tsv"
        elif variant == "cascade":
            rows = run_cascade(
                samples, config_path=args.cascade_config, verbose=args.verbose
            )
            out_name = "cascade.tsv"
        else:
            print(f"ERROR: variante inconnue: {variant}", file=sys.stderr)
            return 2

        for row in rows:
            row["src_text_fr"] = src_map.get(row["id"], "")

        out_path = args.output_dir / out_name
        _write_results_tsv(out_path, rows)
        print(f"Écrit: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
