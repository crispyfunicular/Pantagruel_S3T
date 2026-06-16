"""Tests sauvegarde checkpoint speechLLM (préfixes projecteur / encodeur)."""

from __future__ import annotations

from speechLLM.speechllm_lib import _speechllm_checkpoint_prefixes


def test_checkpoint_prefixes_frozen_encoder() -> None:
    """B1 gelé : seul le projecteur est persisté."""
    prefixes = _speechllm_checkpoint_prefixes({"model": {"freeze_encoder": True}})
    assert "encoder." not in prefixes
    assert "projector." in prefixes


def test_checkpoint_prefixes_unfrozen_encoder() -> None:
    """Encodeur dégelé : les poids encodeur doivent être dans le checkpoint."""
    prefixes = _speechllm_checkpoint_prefixes({"model": {"freeze_encoder": False}})
    assert "encoder." in prefixes
