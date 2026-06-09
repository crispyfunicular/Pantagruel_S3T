"""Tests pour la synchronisation runs/experiments_tracking.csv."""

from __future__ import annotations

import json
from pathlib import Path

from scripts_communs.update_experiments_tracking import (
    DEFAULT_GEMINI_FLASH_INPUT_USD_PER_1M,
    DEFAULT_GEMINI_FLASH_OUTPUT_USD_PER_1M,
    _load_metrics_row,
    upsert_tracking_row,
)


def test_load_gemini_metrics_row_includes_pricing(tmp_path: Path) -> None:
    """Une ligne Gemini doit exposer durée, coût et grille tarifaire."""
    metrics = {
        "pipeline": "gemini_st",
        "run_id": "run_test_gemini",
        "model_id": "gemini-2.5-flash",
        "runtime": {"elapsed_minutes": 12.5},
        "gemini_cost_estimate_usd": {
            "input_per_1m_tokens_usd": 1.0,
            "output_per_1m_tokens_usd": 2.5,
            "total": 0.42,
        },
        "dev": {"bleu": 21.0, "chrf": 40.0, "ter": 70.0, "signature": "sig"},
        "test": {"bleu": 23.0, "chrf": 41.0, "ter": 68.0},
        "config": {
            "experiment": {"lang_pair": "fr-en"},
            "data": {"valid_manifest": "datasets/manifests_sentence/fr-en/valid.tsv"},
        },
    }
    path = tmp_path / "eval" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")

    row = _load_metrics_row(path)
    assert row["pipeline"] == "gemini_st"
    assert row["segment_mode"] == "sentence_like"
    assert row["gemini_duration_min"] == "12.50"
    assert row["gemini_cost_usd"] == "0.4200"
    assert (
        row["gemini_input_usd_per_1m"] == f"{DEFAULT_GEMINI_FLASH_INPUT_USD_PER_1M:.2f}"
    )
    assert (
        row["gemini_output_usd_per_1m"]
        == f"{DEFAULT_GEMINI_FLASH_OUTPUT_USD_PER_1M:.2f}"
    )


def test_load_transformer_metrics_infers_pipeline(tmp_path: Path) -> None:
    """Un run Transformer sans champ pipeline doit être reconnu via le run_id."""
    metrics = {
        "run_id": "run_004_transformer_baseline_utterance_v2",
        "beam_size": 5,
        "gpu_hours": 0.036,
        "dev": {"bleu": 16.84, "chrf": 40.58, "ter": 72.86},
        "test": {"bleu": 16.68, "chrf": 40.92, "ter": 73.17},
        "config": {
            "experiment": {"lang_pair": "fr-en", "seed": 42},
            "data": {"segment_mode": "utterance"},
            "train": {"freeze_encoder_updates": 5000},
            "model": {"encoder_name": "PantagrueLLM/speech-base-1K"},
        },
    }
    path = tmp_path / "eval" / "metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(metrics), encoding="utf-8")

    row = _load_metrics_row(path)
    assert row["pipeline"] == "transformer"
    assert row["segment_mode"] == "utterance"
    assert row["bleu_test"] == "16.68"
    assert row["freeze_updates"] == "5000"
    assert row["seed"] == "42"


def test_upsert_tracking_row_replaces_by_run_id(tmp_path: Path) -> None:
    """Le CSV doit remplacer une ligne existante avec le même run_id."""
    csv_path = tmp_path / "experiments_tracking.csv"
    upsert_tracking_row(
        {"run_id": "run_a", "lang_pair": "fr-en", "bleu_dev": "1.00"},
        tracking_path=csv_path,
    )
    upsert_tracking_row(
        {"run_id": "run_a", "lang_pair": "fr-en", "bleu_dev": "2.00"},
        tracking_path=csv_path,
    )
    text = csv_path.read_text(encoding="utf-8")
    assert text.count("run_a") == 1
    assert "2.00" in text
