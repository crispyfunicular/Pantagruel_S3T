"""Tests du protocole d'évaluation figé (scripts_communs/eval_protocol.py)."""

from __future__ import annotations

import json

from scripts_communs.eval_protocol import (
    EVAL_PROTOCOL_VERSION,
    build_protocol_record,
    score_corpus_metrics,
    write_eval_protocol_artifact,
)


def test_score_corpus_metrics_signature() -> None:
    """BLEU corpus + signature sur paires alignées."""
    preds = ["hello world", "second line"]
    refs = ["hello world", "second reference"]
    scores = score_corpus_metrics(preds, refs)
    assert scores["bleu"] >= 0.0
    assert "tok:13a" in scores["signature"]
    assert scores["chrf"] >= 0.0
    assert scores["ter"] >= 0.0


def test_score_corpus_metrics_length_mismatch() -> None:
    """Erreur si preds et refs ont des longueurs différentes."""
    try:
        score_corpus_metrics(["a"], [])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_write_protocol_artifact(tmp_path) -> None:
    """protocol.json contient la version figée."""
    eval_dir = tmp_path / "eval"
    record = build_protocol_record(
        pipeline="gemini_st",
        lang_pair="fr-en",
        run_id="run_test",
        segment_mode="sentence_like",
        config_path="3_Gemini/configs/fr-en/gemini_flash_sentence.yaml",
        decode={"temperature": 0.0, "max_output_tokens": 256},
        sacrebleu_signatures={"dev": "sig-dev", "test": "sig-test"},
        n_segments={"dev": 10, "test": 20},
    )
    path = write_eval_protocol_artifact(eval_dir, record)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["eval_protocol_version"] == EVAL_PROTOCOL_VERSION
    assert payload["pipeline"] == "gemini_st"
    assert payload["n_segments"]["test"] == 20
