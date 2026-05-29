"""Tests unitaires speechLLM (sans téléchargement Hugging Face)."""

from __future__ import annotations

import torch
from scripts_communs.st_common import ManifestSample
from speechLLM.speechllm_common import (
    IGNORE_INDEX,
    collate_speechllm_batch,
    downsample_encoder_states,
)


def test_downsample_encoder_states_k5() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)
    mask = torch.ones((1, 8), dtype=torch.long)
    down, down_mask = downsample_encoder_states(hidden, mask, k=5)
    assert down.shape == (1, 1, 15)
    assert down_mask.shape == (1, 1)
    assert down_mask.item() == 1


def test_downsample_k1_is_identity() -> None:
    hidden = torch.randn(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    out, out_mask = downsample_encoder_states(hidden, mask, k=1)
    assert torch.equal(out, hidden)
    assert torch.equal(out_mask, mask)


def test_downsample_trims_incomplete_tail() -> None:
    hidden = torch.ones((1, 7, 2))
    mask = torch.ones((1, 7), dtype=torch.long)
    down, down_mask = downsample_encoder_states(hidden, mask, k=3)
    assert down.shape == (1, 2, 6)
    assert down_mask.shape == (1, 2)


def test_ignore_index_constant() -> None:
    assert IGNORE_INDEX == -100


def test_collate_speechllm_batch_empty() -> None:
    batch: list[ManifestSample] = []
    result = collate_speechllm_batch(batch, sample_rate=16000)
    assert result["input_values"].shape == (0, 0)
    assert result["target_texts"] == []
