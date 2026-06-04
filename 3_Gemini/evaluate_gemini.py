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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Gemini.gemini_common import (
    DEFAULT_GEMINI_MODEL_ID,
    DEFAULT_PROMPT,
    GeminiRequest,
    GeminiTranslationResult,
    GeminiUsage,
    MissingGeminiApiKeyError,
    create_gemini_client,
    translate_audio_with_metadata,
)
from scripts_communs.eval_protocol import (
    build_protocol_record,
    score_corpus_metrics,
    write_eval_protocol_artifact,
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
    input_per_1m_tokens_usd = float(
        deep_get(config, "pricing.input_per_1m_tokens_usd", 0.0)
    )
    output_per_1m_tokens_usd = float(
        deep_get(config, "pricing.output_per_1m_tokens_usd", 0.0)
    )
    fixed_per_request_usd = float(
        deep_get(config, "pricing.fixed_per_request_usd", 0.0)
    )

    if dry_run:
        print("[dry-run] gemini evaluate:")
        print(f"  run_dir:   {run_dir}")
        print(f"  model:     {model_id}")
        print(f"  prompt:    {prompt}")
        print(f"  limit:     {limit if limit > 0 else 'none'}")
        print("  pricing:   input/output per 1M tokens + fixed/request")
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

    def _usage_to_dict(usage: GeminiUsage) -> dict[str, int]:
        """Convertir ``GeminiUsage`` optionnel en dict d'entiers cumulables."""
        return {
            "prompt_tokens": int(usage.prompt_tokens or 0),
            "candidate_tokens": int(usage.candidate_tokens or 0),
            "total_tokens": int(usage.total_tokens or 0),
        }

    def _estimate_cost_usd(usage_stats: dict[str, int], requests: int) -> float:
        """Estimer le coût USD à partir des tokens et du coût fixe par requête."""
        return (
            (usage_stats["prompt_tokens"] / 1_000_000.0) * input_per_1m_tokens_usd
            + (usage_stats["candidate_tokens"] / 1_000_000.0) * output_per_1m_tokens_usd
            + requests * fixed_per_request_usd
        )

    def translate_split(
        manifest_path: Path,
    ) -> tuple[list[str], list[str], list[dict], dict[str, Any]]:
        """Traduire un split TSV et renvoyer hypothèses, références, et erreurs."""
        preds: list[str] = []
        refs: list[str] = []
        failures: list[dict] = []
        split_usage = {"prompt_tokens": 0, "candidate_tokens": 0, "total_tokens": 0}
        split_requests = 0
        split_retries = 0
        split_call_seconds = 0.0
        samples = read_manifest(manifest_path)
        if limit > 0:
            samples = samples[:limit]
        for sample in samples:
            refs.append(sample.target_text)
            last_error: str | None = None
            hyp = ""
            for attempt in range(max(1, max_retries + 1)):
                split_requests += 1
                started = time.perf_counter()
                try:
                    result: GeminiTranslationResult = translate_audio_with_metadata(
                        client=client,
                        request=request,
                        audio_path=sample.audio_path,
                    )
                    split_call_seconds += time.perf_counter() - started
                    usage_dict = _usage_to_dict(result.usage)
                    split_usage["prompt_tokens"] += usage_dict["prompt_tokens"]
                    split_usage["candidate_tokens"] += usage_dict["candidate_tokens"]
                    split_usage["total_tokens"] += usage_dict["total_tokens"]
                    hyp = result.text
                    break
                except Exception as exc:  # noqa: BLE001 — baseline robuste (API/network)
                    split_call_seconds += time.perf_counter() - started
                    last_error = f"{type(exc).__name__}: {exc}"
                    split_retries += 1
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
        return (
            preds,
            refs,
            failures,
            {
                "samples": len(samples),
                "requests": split_requests,
                "retries": split_retries,
                "call_seconds": split_call_seconds,
                "usage": split_usage,
            },
        )

    eval_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Evaluating Gemini model: {model_id}")
        print(f"Eval dir: {eval_dir}")

    eval_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    dev_preds, dev_refs, dev_failures, dev_runtime = translate_split(valid_manifest)
    test_preds, test_refs, test_failures, test_runtime = translate_split(test_manifest)

    dev_scores = score_corpus_metrics(dev_preds, dev_refs)
    test_scores = score_corpus_metrics(test_preds, test_refs)

    from scripts_communs.export_eval_review import write_eval_review_artifacts

    write_eval_review_artifacts(eval_dir, "dev", valid_manifest, dev_preds)
    write_eval_review_artifacts(eval_dir, "test", test_manifest, test_preds)
    elapsed_seconds = time.perf_counter() - eval_started
    finished_utc = datetime.now(timezone.utc).isoformat()

    total_usage = {
        "prompt_tokens": dev_runtime["usage"]["prompt_tokens"]
        + test_runtime["usage"]["prompt_tokens"],
        "candidate_tokens": dev_runtime["usage"]["candidate_tokens"]
        + test_runtime["usage"]["candidate_tokens"],
        "total_tokens": dev_runtime["usage"]["total_tokens"]
        + test_runtime["usage"]["total_tokens"],
    }
    total_requests = int(dev_runtime["requests"]) + int(test_runtime["requests"])
    cost_dev_usd = _estimate_cost_usd(
        dev_runtime["usage"], int(dev_runtime["requests"])
    )
    cost_test_usd = _estimate_cost_usd(
        test_runtime["usage"],
        int(test_runtime["requests"]),
    )
    cost_total_usd = cost_dev_usd + cost_test_usd

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
            "runtime": {
                "started_utc": started_utc,
                "finished_utc": finished_utc,
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_seconds / 60.0,
                "api_call_seconds": float(dev_runtime["call_seconds"])
                + float(test_runtime["call_seconds"]),
            },
            "gemini_usage": {
                "dev": dev_runtime,
                "test": test_runtime,
                "total": {
                    "samples": int(dev_runtime["samples"])
                    + int(test_runtime["samples"]),
                    "requests": total_requests,
                    "retries": int(dev_runtime["retries"])
                    + int(test_runtime["retries"]),
                    "usage": total_usage,
                },
            },
            "gemini_cost_estimate_usd": {
                "input_per_1m_tokens_usd": input_per_1m_tokens_usd,
                "output_per_1m_tokens_usd": output_per_1m_tokens_usd,
                "fixed_per_request_usd": fixed_per_request_usd,
                "dev": cost_dev_usd,
                "test": cost_test_usd,
                "total": cost_total_usd,
            },
            "dev": dev_scores,
            "test": test_scores,
            "failures": {
                "dev": dev_failures,
                "test": test_failures,
            },
        },
    )

    segment_mode = str(deep_get(config, "data.segment_mode", "utterance"))
    lang_pair = str(deep_get(config, "experiment.lang_pair", "fr-en"))
    write_eval_protocol_artifact(
        eval_dir,
        build_protocol_record(
            pipeline="gemini_st",
            lang_pair=lang_pair,
            run_id=run_id,
            segment_mode=segment_mode,
            config_path=config_path,
            decode={
                "model_id": model_id,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
                "prompt": prompt,
            },
            sacrebleu_signatures={
                "dev": dev_scores["signature"],
                "test": test_scores["signature"],
            },
            n_segments={"dev": len(dev_preds), "test": len(test_preds)},
            extra={"limit": limit, "max_retries": max_retries},
        ),
    )

    print("Gemini evaluation complete.")
    print(f"  BLEU dev:  {dev_scores['bleu']:.2f}")
    print(f"  BLEU test: {test_scores['bleu']:.2f}")
    print(f"  Duration:  {elapsed_seconds / 60.0:.2f} min")
    print(f"  Cost est:  ${cost_total_usd:.6f}")
    print(f"  Eval dir:  {eval_dir}")

    try:
        from scripts_communs.update_experiments_tracking import sync_run_from_metrics

        if sync_run_from_metrics(run_dir):
            print(f"  Tracking:  runs/experiments_tracking.csv (run_id={run_id})")
    except OSError as exc:
        print(f"  WARNING: tracking CSV not updated: {exc}", file=sys.stderr)

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
