#!/usr/bin/env python3
"""
Évaluation Gemini ST — décodage valid/test et métriques SacreBLEU signées.

Cette étape produit des artefacts compatibles avec les autres pistes :
`runs/<langpair>/<run_id>/eval/{dev,test}_predictions.txt`, `sacrebleu_*.txt`,
et `metrics.json`.

Entrées :
- config YAML (chemins manifests, prompt, modèle Gemini) ;
- `--run-id` (répertoire `runs/...`).

Dépendances :
- `google-genai` (client Gemini)
- `sacrebleu` (métriques)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sacrebleu
from Gemini.gemini_common import (
    DEFAULT_GEMINI_MODEL_ID,
    DEFAULT_PROMPT,
    GeminiRequest,
    MissingGeminiApiKeyError,
    create_gemini_client,
    translate_audio,
)
from speechLLM.speechllm_common import (
    PROJECT_ROOT,
    deep_get,
    load_yaml_config,
    read_manifest,
    resolve_run_dir,
    resolve_speechllm_config_path,
    write_json,
)


def score_with_sacrebleu(preds: list[str], refs: list[str]) -> dict[str, Any]:
    """Calculer BLEU, CHRF, TER corpus et la signature SacreBLEU."""
    bleu_metric = sacrebleu.metrics.BLEU()
    chrf_metric = sacrebleu.metrics.CHRF()
    ter_metric = sacrebleu.metrics.TER()
    bleu = bleu_metric.corpus_score(preds, [refs])
    chrf = chrf_metric.corpus_score(preds, [refs])
    ter = ter_metric.corpus_score(preds, [refs])
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "ter": float(ter.score),
        "signature": str(bleu_metric.get_signature()),
        "bleu_text": str(bleu),
        "chrf_text": str(chrf),
        "ter_text": str(ter),
    }


def run_evaluate_gemini(
    *,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
    limit: int,
    max_retries: int,
    dry_run: bool,
    verbose: bool,
) -> int:
    """Évaluer valid et test via Gemini (API) et écrire les artefacts SacreBLEU."""
    config_path = resolve_speechllm_config_path(config_path)
    config = load_yaml_config(config_path)
    run_dir = resolve_run_dir(config, run_id=run_id, output_dir_override=output_dir)
    eval_dir = run_dir / "eval"

    valid_manifest = PROJECT_ROOT / str(deep_get(config, "data.valid_manifest"))
    test_manifest = PROJECT_ROOT / str(deep_get(config, "data.test_manifest"))
    prompt = str(deep_get(config, "prompt.template", DEFAULT_PROMPT))
    model_id = str(deep_get(config, "model.gemini_id", DEFAULT_GEMINI_MODEL_ID))
    temperature = float(deep_get(config, "decode.temperature", 0.0))
    max_output_tokens = int(deep_get(config, "decode.max_output_tokens", 256))

    if dry_run:
        print("[dry-run] gemini evaluate:")
        print(f"  run_dir:   {run_dir}")
        print(f"  model:     {model_id}")
        print(f"  prompt:    {prompt}")
        print(f"  limit:     {limit if limit > 0 else 'none'}")
        return 0

    if not valid_manifest.is_file() or not test_manifest.is_file():
        print("ERROR: missing valid/test manifest", file=sys.stderr)
        return 2

    try:
        client = create_gemini_client()
    except MissingGeminiApiKeyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    request = GeminiRequest(
        model_id=model_id,
        prompt=prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )

    def translate_split(manifest_path: Path) -> tuple[list[str], list[str], list[dict]]:
        """Traduire un split TSV et renvoyer hypothèses, références, et erreurs."""
        preds: list[str] = []
        refs: list[str] = []
        failures: list[dict] = []
        samples = read_manifest(manifest_path)
        if limit > 0:
            samples = samples[:limit]
        for sample in samples:
            refs.append(sample.target_text)
            last_error: str | None = None
            hyp = ""
            for attempt in range(max(1, max_retries + 1)):
                try:
                    hyp = translate_audio(
                        client=client,
                        request=request,
                        audio_path=sample.audio_path,
                    )
                    break
                except Exception as exc:  # noqa: BLE001 — baseline robuste (API/network)
                    last_error = f"{type(exc).__name__}: {exc}"
                    if verbose:
                        print(
                            f"[gemini] retry {attempt}/{max_retries} for {sample.sample_id}: {last_error}",
                            file=sys.stderr,
                        )
            if not hyp and last_error is not None:
                failures.append(
                    {
                        "id": sample.sample_id,
                        "audio": str(sample.audio_path),
                        "error": last_error,
                    }
                )
            preds.append(hyp)
        return preds, refs, failures

    eval_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Evaluating Gemini model: {model_id}")
        print(f"Eval dir: {eval_dir}")

    dev_preds, dev_refs, dev_failures = translate_split(valid_manifest)
    test_preds, test_refs, test_failures = translate_split(test_manifest)

    dev_scores = score_with_sacrebleu(dev_preds, dev_refs)
    test_scores = score_with_sacrebleu(test_preds, test_refs)

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
            "pipeline": "gemini_st",
            "run_id": run_id,
            "config": str(config_path.resolve()),
            "model_id": model_id,
            "prompt": prompt,
            "decode": {
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "max_retries": max_retries,
                "limit": limit,
            },
            "dev": dev_scores,
            "test": test_scores,
            "failures": {
                "dev": dev_failures,
                "test": test_failures,
            },
        },
    )

    print("Gemini evaluation complete.")
    print(f"  BLEU dev:  {dev_scores['bleu']:.2f}")
    print(f"  BLEU test: {test_scores['bleu']:.2f}")
    print(f"  Eval dir:  {eval_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape `evaluate-gemini`."""
    parser = argparse.ArgumentParser(description="Gemini ST — évaluation SacreBLEU")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par `2_speechLLM/pipeline.py evaluate-gemini`."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        print("ERROR: --config is required", file=sys.stderr)
        return 2
    run_id = getattr(args, "run_id", None)
    if run_id is None:
        print("ERROR: --run-id is required", file=sys.stderr)
        return 2
    return run_evaluate_gemini(
        config_path=config_path,
        run_id=run_id,
        output_dir=getattr(args, "output_dir", None),
        limit=int(getattr(args, "limit", 0)),
        max_retries=int(getattr(args, "max_retries", 2)),
        dry_run=bool(getattr(args, "dry_run", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    return run_from_namespace(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
