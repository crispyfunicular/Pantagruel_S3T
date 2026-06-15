"""
Protocole d'évaluation S3T — version figée et artefacts reproductibles.

Ce module est la **source de vérité machine** pour la méthodologie documentée dans
``docs/protocole_evaluation.md``. Toute modification des règles ci-dessous impose
une **nouvelle version** de ``EVAL_PROTOCOL_VERSION`` et une re-évaluation des runs
comparables.

Sorties typiques : ``runs/<langpair>/<run_id>/eval/protocol.json`` écrit par chaque
étape ``evaluate`` (Transformer, speechLLM, Gemini, cascade).
"""

from __future__ import annotations

import importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sacrebleu

# ---------------------------------------------------------------------------
# Version figée — incrémenter si le protocole change (décodage, métrique, splits).
# ---------------------------------------------------------------------------
EVAL_PROTOCOL_VERSION = "2026-06-12-v2"
EVAL_PROTOCOL_DOC = "docs/protocole_evaluation.md"

SACREBLEU_MIN_VERSION = "2.3.0"

# Métrique principale pour comparaison inter-variantes et Table 8 indicative.
PRIMARY_METRIC = "bleu_corpus"

# Spécifications de décodage **telles qu'implémentées** dans le dépôt (v1).
# ``decode_target`` = objectif article / PRD ; ``decode_implemented`` = code actuel.
PIPELINE_DECODE_SPECS: dict[str, dict[str, Any]] = {
    "transformer_st": {
        "decode_target": {"mode": "beam", "beam_size": 5, "max_len_b": 128},
        "decode_implemented": {
            "mode": "beam_or_greedy",
            "beam_size_from": "decode.beam_size YAML ou --beam-size CLI",
            "note": "1_Transformer/5_evaluate.py : decode_batch (greedy si beam<=1).",
        },
        "text": "spm_decode_then_strip_special_tokens",
        "checkpoint": "checkpoints/best.pt (meilleur SacreBLEU dev en entraînement)",
    },
    "speechllm": {
        "decode_target": {"beam_size": 1, "max_new_tokens": 48},
        "decode_implemented": {"beam_size": 1, "max_new_tokens": 48},
        "text": "manifest_tgt_text_raw_utf8",
        "checkpoint": "checkpoints/best.pt (projecteur ; encodeur si unfreeze)",
    },
    "gemini_st": {
        "decode_target": {
            "temperature": 0.0,
            "max_output_tokens": 256,
        },
        "decode_implemented": {
            "temperature": 0.0,
            "max_output_tokens": 256,
        },
        "text": "manifest_tgt_text_raw_utf8",
        "checkpoint": "none_api",
    },
    "cascade_asr_mt": {
        "decode_target": {
            "asr": "whisper (language=fr)",
            "mt_max_length": 256,
        },
        "decode_implemented": {
            "asr": "whisper (language=fr)",
            "mt_max_length": 256,
        },
        "text": "manifest_tgt_text_raw_utf8",
        "checkpoint": "none_pretrained",
    },
}


def sacrebleu_package_version() -> str:
    """Version installée du paquet ``sacrebleu`` (verrouillage reproductibilité)."""
    try:
        return importlib.metadata.version("sacrebleu")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def score_corpus_metrics(preds: list[str], refs: list[str]) -> dict[str, Any]:
    """
    Calculer BLEU, chrF et TER corpus avec les paramètres par défaut SacreBLEU.

    Aucune normalisation NFKC/minuscules n'est appliquée aux chaînes : hypothèses et
    références sont comparées **telles qu'écrites** dans les fichiers de prédictions.

    Paramètres :
        preds : Hypothèses (une chaîne par segment, ordre manifest).
        refs : Références cible alignées (même cardinalité).

    Retour :
        Dict avec scores numériques, lignes formatées et ``signature`` BLEU.
    """
    if len(preds) != len(refs):
        msg = f"pred/ref length mismatch: {len(preds)} vs {len(refs)}"
        raise ValueError(msg)
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


def build_protocol_record(
    *,
    pipeline: str,
    lang_pair: str,
    run_id: str,
    segment_mode: str,
    config_path: str | Path,
    decode: dict[str, Any],
    sacrebleu_signatures: dict[str, str],
    n_segments: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Construire le dict ``protocol.json`` pour un run évalué.

    Paramètres :
        pipeline : Clé ``PIPELINE_DECODE_SPECS`` (ex. ``transformer_st``).
        lang_pair : Paire linguistique (ex. ``fr-en``).
        run_id : Identifiant de run.
        segment_mode : ``utterance`` ou ``sentence_like``.
        config_path : Chemin YAML résolu.
        decode : Paramètres de décodage effectivement utilisés.
        sacrebleu_signatures : Signatures par split (``dev``, ``test``).
        n_segments : Nombre de lignes scorées par split.
        extra : Champs optionnels (modèle API, limit smoke, etc.).

    Retour :
        Dict sérialisable JSON.
    """
    spec = PIPELINE_DECODE_SPECS.get(pipeline, {})
    record: dict[str, Any] = {
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        "eval_protocol_doc": EVAL_PROTOCOL_DOC,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "sacrebleu_package_version": sacrebleu_package_version(),
        "sacrebleu_min_version": SACREBLEU_MIN_VERSION,
        "pipeline": pipeline,
        "lang_pair": lang_pair,
        "run_id": run_id,
        "segment_mode": segment_mode,
        "config": str(Path(config_path).resolve()),
        "splits": {
            "dev": {
                "manifest_role": "valid",
                "predictions_file": "dev_predictions.txt",
                "references_source": "manifest_tgt_text",
            },
            "test": {
                "manifest_role": "test",
                "predictions_file": "test_predictions.txt",
                "references_source": "manifest_tgt_text",
            },
        },
        "text_normalization": "none",
        "decode": decode,
        "pipeline_spec": spec,
        "sacrebleu_signatures": sacrebleu_signatures,
    }
    if n_segments is not None:
        record["n_segments"] = n_segments
    if extra:
        record["extra"] = extra
    return record


def write_eval_protocol_artifact(
    eval_dir: Path,
    record: dict[str, Any],
) -> Path:
    """
    Écrire ``eval/protocol.json`` (protocole figé du run).

    Paramètres :
        eval_dir : Répertoire ``runs/.../eval/``.
        record : Payload de ``build_protocol_record``.

    Retour :
        Chemin du fichier écrit.
    """
    from scripts_communs.config_utils import write_json

    eval_dir.mkdir(parents=True, exist_ok=True)
    path = eval_dir / "protocol.json"
    write_json(path, record)
    return path
