#!/usr/bin/env python3
"""
Évaluation baselines ST open source — décodage valid/test et métriques SacreBLEU signées.

Produit des artefacts compatibles avec les autres pistes :
``runs/<langpair>/<run_id>/eval/{dev,test}_predictions.txt``, ``sacrebleu_*.txt``,
et ``metrics.json`` (champ ``pipeline = open_baselines_st``).

Entrées :
- config YAML (manifests, modèle open) ;
- ``--run-id`` (répertoire ``runs/...``).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from OpenBaselines.open_common import (
    EXIT_CONFIG,
    EXIT_NOT_IMPLEMENTED,
    EXIT_SUCCESS,
    OpenPipelineNotReadyError,
    load_open_settings,
    open_translate_audio,
    resolve_open_config_path,
)
from scripts_communs.eval_protocol import (
    build_protocol_record,
    score_corpus_metrics,
    write_eval_protocol_artifact,
)
from speechLLM.speechllm_lib import (
    PROJECT_ROOT,
    deep_get,
    load_yaml_config,
    read_manifest,
    resolve_run_dir,
    write_json,
)


def run_evaluate_open(
    *,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
    limit: int,
    dry_run: bool,
    verbose: bool,
) -> int:
    """
    Évaluer valid et test via un modèle ST open et écrire les artefacts SacreBLEU.

    Paramètres :
        config_path : YAML d'expérience sous ``6_open_baselines/configs/``.
        run_id : Identifiant de run (sous-répertoire ``runs/``).
        output_dir : Override optionnel du répertoire de run.
        limit : Nombre max de segments par split (0 = illimité).
        dry_run : Afficher le plan sans inférence ni écriture ``eval/``.
        verbose : Logs détaillés sur stderr.

    Retour :
        Code de sortie processus (0, 2 ou 3).
    """
    config_path = resolve_open_config_path(config_path)
    config = load_yaml_config(config_path)
    settings = load_open_settings(config)
    run_dir = resolve_run_dir(config, run_id=run_id, output_dir_override=output_dir)
    eval_dir = run_dir / "eval"

    valid_manifest = PROJECT_ROOT / str(deep_get(config, "data.valid_manifest"))
    test_manifest = PROJECT_ROOT / str(deep_get(config, "data.test_manifest"))
    segment_mode = str(deep_get(config, "data.segment_mode", "utterance"))

    if dry_run:
        print("[dry-run] open baselines evaluate:")
        print(f"  run_dir:       {run_dir}")
        print(f"  eval_dir:      {eval_dir}")
        print(f"  segment_mode:  {segment_mode}")
        print(f"  valid_manifest:{valid_manifest}")
        print(f"  test_manifest: {test_manifest}")
        print(f"  model.type:    {settings.model_type}")
        print(f"  model.model_id:{settings.model_id}")
        print(f"  limit:         {limit if limit > 0 else 'none'}")
        return EXIT_SUCCESS

    if not valid_manifest.is_file() or not test_manifest.is_file():
        print("ERROR: missing valid/test manifest", file=sys.stderr)
        return EXIT_CONFIG

    def translate_split(manifest_path: Path) -> tuple[list[str], list[str], list[dict]]:
        """Traduire un split TSV et collecter hypothèses / références."""
        preds: list[str] = []
        refs: list[str] = []
        failures: list[dict] = []
        samples = read_manifest(manifest_path)
        if limit > 0:
            samples = samples[:limit]
        for sample in samples:
            refs.append(sample.target_text)
            try:
                hyp = open_translate_audio(sample.audio_path, settings)
            except OpenPipelineNotReadyError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return [], [], [{"fatal": str(exc)}]
            except Exception as exc:  # noqa: BLE001 — journaliser et continuer
                failures.append(
                    {
                        "id": sample.sample_id,
                        "audio": str(sample.audio_path),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                hyp = ""
            preds.append(hyp)
        return preds, refs, failures

    if verbose:
        print(
            f"Evaluating open baseline: type={settings.model_type}, "
            f"model={settings.model_id}"
        )
        print(f"Eval dir: {eval_dir}")

    dev_preds, dev_refs, dev_failures = translate_split(valid_manifest)
    if dev_failures and dev_failures[0].get("fatal"):
        return EXIT_NOT_IMPLEMENTED

    test_preds, test_refs, test_failures = translate_split(test_manifest)
    if test_failures and test_failures[0].get("fatal"):
        return EXIT_NOT_IMPLEMENTED

    eval_dir.mkdir(parents=True, exist_ok=True)
    dev_scores = score_corpus_metrics(dev_preds, dev_refs)
    test_scores = score_corpus_metrics(test_preds, test_refs)

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
            "pipeline": "open_baselines_st",
            "run_id": run_id,
            "config": str(config_path.resolve()),
            "segment_mode": segment_mode,
            "model": {
                "type": settings.model_type,
                "model_id": settings.model_id,
                "src_lang": settings.src_lang,
                "tgt_lang": settings.tgt_lang,
                "device": settings.device,
                "dtype": settings.dtype,
            },
            "decode": {"max_new_tokens": settings.max_new_tokens, "limit": limit},
            "dev": dev_scores,
            "test": test_scores,
            "failures": {"dev": dev_failures, "test": test_failures},
        },
    )

    lang_pair = str(deep_get(config, "experiment.lang_pair", "fr-en"))
    write_eval_protocol_artifact(
        eval_dir,
        build_protocol_record(
            pipeline="open_baselines_st",
            lang_pair=lang_pair,
            run_id=run_id,
            segment_mode=segment_mode,
            config_path=config_path,
            decode={
                "model_type": settings.model_type,
                "model_id": settings.model_id,
                "src_lang": settings.src_lang,
                "tgt_lang": settings.tgt_lang,
                "max_new_tokens": settings.max_new_tokens,
                "limit": limit,
            },
            sacrebleu_signatures={
                "dev": dev_scores["signature"],
                "test": test_scores["signature"],
            },
            n_segments={"dev": len(dev_preds), "test": len(test_preds)},
            extra={
                "failures_dev": len(dev_failures),
                "failures_test": len(test_failures),
            },
        ),
    )

    print("Open baselines evaluation complete.")
    print(f"  BLEU dev:  {dev_scores['bleu']:.2f}")
    print(f"  BLEU test: {test_scores['bleu']:.2f}")
    if dev_failures or test_failures:
        print(
            f"  WARNING: {len(dev_failures)} échecs dev, "
            f"{len(test_failures)} échecs test (hypothèses vides → BLEU 0)"
        )
    print(f"  Eval dir:  {eval_dir}")
    return EXIT_SUCCESS


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape ``evaluate`` open baselines."""
    parser = argparse.ArgumentParser(
        description="Open baselines ST — évaluation SacreBLEU"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée partagé (CLI directe ou routeur ``6_open_baselines/pipeline.py``)."""
    return run_evaluate_open(
        config_path=args.config,
        run_id=args.run_id,
        output_dir=args.output_dir,
        limit=args.limit,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> int:
    """``main`` CLI autonome."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
