#!/usr/bin/env python3
"""
Évaluation speechLLM — décodage valid/test et métriques SacreBLEU signées.

Entrées : config YAML, checkpoint ``best.pt`` (projecteur ; + encodeur si ``freeze_encoder: false``).
Sorties : ``runs/.../eval/`` (prédictions, sacrebleu_*.txt, metrics.json).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from scripts_communs.eval_protocol import (
    build_protocol_record,
    score_corpus_metrics,
    write_eval_protocol_artifact,
)
from speechLLM.speechllm_lib import (
    PROJECT_ROOT,
    SpeechLLMModel,
    collate_speechllm_batch,
    deep_get,
    load_projector_checkpoint,
    load_speechllm_checkpoint,
    load_speechllm_from_config,
    load_yaml_config,
    read_manifest,
    resolve_run_dir,
    resolve_speechllm_config_path,
    write_json,
)
from torch.utils.data import DataLoader


def decode_manifest(
    *,
    model: SpeechLLMModel,
    loader: DataLoader,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
) -> tuple[list[str], list[str]]:
    """Décoder un manifest et renvoyer hypothèses / références texte brut."""
    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            hyps = model.generate_text_batch(
                batch["input_values"].to(device),
                batch["attention_mask"].to(device),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            predictions.extend(hyps)
            references.extend(batch["target_texts"])
    return predictions, references


def run_evaluate(
    *,
    config_path: Path,
    run_id: str,
    checkpoint: Path | None,
    beam_size: int,
    output_dir: Path | None,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """Évaluer valid et test pour un run speechLLM."""
    config_path = resolve_speechllm_config_path(config_path)
    config = load_yaml_config(config_path)
    run_dir = resolve_run_dir(config, run_id=run_id, output_dir_override=output_dir)
    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    checkpoint_path = checkpoint or (checkpoints_dir / "best.pt")

    valid_manifest = PROJECT_ROOT / str(deep_get(config, "data.valid_manifest"))
    test_manifest = PROJECT_ROOT / str(deep_get(config, "data.test_manifest"))
    sample_rate = int(deep_get(config, "data.sample_rate", 16000))
    batch_size = int(deep_get(config, "train.batch_size", 2))
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

    if dry_run:
        print("[dry-run] speechLLM evaluate:")
        print(f"  run_dir:    {run_dir}")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  beams:      {num_beams}")
        return 0

    if not valid_manifest.is_file() or not test_manifest.is_file():
        print("ERROR: missing valid/test manifest", file=sys.stderr)
        return 2
    if not checkpoint_path.is_file():
        print(f"ERROR: missing checkpoint: {checkpoint_path}", file=sys.stderr)
        return 2

    start_wall_s = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()

    payload = load_speechllm_checkpoint(checkpoint_path)
    ckpt_config = payload.get("config", config)
    if not isinstance(ckpt_config, dict):
        ckpt_config = config

    device = torch.device(
        "cpu" if prefer_cpu or not torch.cuda.is_available() else "cuda"
    )
    model = load_speechllm_from_config(ckpt_config, device=device)
    load_projector_checkpoint(model, payload)

    def collate_fn(batch: list) -> dict:
        """Collate DataLoader : audio paddé + textes cibles bruts (speechLLM)."""
        return collate_speechllm_batch(batch, sample_rate=sample_rate)

    valid_loader = DataLoader(
        read_manifest(valid_manifest),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        read_manifest(test_manifest),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Evaluating speechLLM checkpoint: {checkpoint_path}")

    dev_preds, dev_refs = decode_manifest(
        model=model,
        loader=valid_loader,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    test_preds, test_refs = decode_manifest(
        model=model,
        loader=test_loader,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )

    dev_scores = score_corpus_metrics(dev_preds, dev_refs)
    test_scores = score_corpus_metrics(test_preds, test_refs)

    from scripts_communs.export_eval_review import write_eval_review_artifacts

    write_eval_review_artifacts(
        eval_dir,
        "dev",
        PROJECT_ROOT / str(deep_get(config, "data.valid_manifest")),
        dev_preds,
    )
    write_eval_review_artifacts(
        eval_dir,
        "test",
        PROJECT_ROOT / str(deep_get(config, "data.test_manifest")),
        test_preds,
    )

    (eval_dir / "dev_predictions.txt").write_text(
        "\n".join(dev_preds) + ("\n" if dev_preds else ""),
        encoding="utf-8",
    )
    (eval_dir / "test_predictions.txt").write_text(
        "\n".join(test_preds) + ("\n" if test_preds else ""),
        encoding="utf-8",
    )
    (eval_dir / "sacrebleu_dev.txt").write_text(
        "\n".join(
            [
                f"BLEU = {dev_scores['bleu']:.2f}",
                dev_scores["bleu_text"],
                f"CHRF = {dev_scores['chrf']:.2f}",
                dev_scores["chrf_text"],
                f"TER = {dev_scores['ter']:.2f}",
                dev_scores["ter_text"],
                f"Signature: {dev_scores['signature']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "sacrebleu_test.txt").write_text(
        "\n".join(
            [
                f"BLEU = {test_scores['bleu']:.2f}",
                test_scores["bleu_text"],
                f"CHRF = {test_scores['chrf']:.2f}",
                test_scores["chrf_text"],
                f"TER = {test_scores['ter']:.2f}",
                test_scores["ter_text"],
                f"Signature: {test_scores['signature']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    write_json(
        eval_dir / "metrics.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "start_timestamp_utc": start_utc,
            "end_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": float(time.time() - start_wall_s),
            "gpu_hours": float(time.time() - start_wall_s) / 3600.0
            if device.type == "cuda"
            else 0.0,
            "estimated_cost_usd": (
                float(time.time() - start_wall_s)
                / 3600.0
                * float(deep_get(config, "cost.usd_per_gpu_hour", 0.0))
                if device.type == "cuda"
                else 0.0
            ),
            "pipeline": "speechllm",
            "run_id": run_id,
            "checkpoint": str(checkpoint_path.resolve()),
            "beam_size": num_beams,
            "device": str(device),
            "config": config,
            "dev": dev_scores,
            "test": test_scores,
        },
    )

    segment_mode = str(
        deep_get(
            config,
            "data.segment_mode",
            "sentence_like"
            if "manifests_sentence" in str(deep_get(config, "data.valid_manifest", ""))
            else "utterance",
        )
    )
    lang_pair = str(deep_get(config, "experiment.lang_pair", "fr-en"))
    write_eval_protocol_artifact(
        eval_dir,
        build_protocol_record(
            pipeline="speechllm",
            lang_pair=lang_pair,
            run_id=run_id,
            segment_mode=segment_mode,
            config_path=config_path,
            decode={
                "beam_size": num_beams,
                "max_new_tokens": max_new_tokens,
                "prompt": prompt,
            },
            sacrebleu_signatures={
                "dev": dev_scores["signature"],
                "test": test_scores["signature"],
            },
            n_segments={"dev": len(dev_preds), "test": len(test_preds)},
        ),
    )

    print("speechLLM evaluation complete.")
    print(f"  BLEU dev:  {dev_scores['bleu']:.2f}")
    print(f"  BLEU test: {test_scores['bleu']:.2f}")
    print(f"  Eval dir:  {eval_dir}")

    try:
        from scripts_communs.update_experiments_tracking import sync_run_from_metrics

        if sync_run_from_metrics(run_dir):
            print(f"  Tracking:  runs/experiments_tracking.csv (run_id={run_id})")
    except OSError as exc:
        print(f"WARNING: tracking CSV not updated: {exc}", file=sys.stderr)

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape `evaluate` speechLLM."""
    parser = argparse.ArgumentParser(description="speechLLM — évaluation SacreBLEU")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--beam-size", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefer-cpu", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par `2_speechLLM/pipeline.py evaluate`."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        print("ERROR: --config is required", file=sys.stderr)
        return 2
    run_id = getattr(args, "run_id", None)
    if run_id is None:
        print("ERROR: --run-id is required", file=sys.stderr)
        return 2
    return run_evaluate(
        config_path=config_path,
        run_id=run_id,
        checkpoint=getattr(args, "checkpoint", None),
        beam_size=getattr(args, "beam_size", 0),
        output_dir=getattr(args, "output_dir", None),
        dry_run=getattr(args, "dry_run", False),
        verbose=getattr(args, "verbose", False),
        prefer_cpu=getattr(args, "prefer_cpu", False),
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
