"""Tests décodage greedy / beam search (st_common)."""

from __future__ import annotations

import torch
import torch.nn as nn
from scripts_communs.st_common import (
    beam_decode_batch,
    decode_batch,
    greedy_decode_batch,
)


class _StubDecodeModel(nn.Module):
    """Modèle minimal : logits favorisent toujours le token 1, puis EOS au 3e pas."""

    def __init__(self, *, vocab_size: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

    def encode(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        batch = input_values.size(0)
        return torch.zeros(batch, 1, self.hidden_dim, device=input_values.device)

    def decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            fill_value=-5.0,
            device=tokens.device,
        )
        # Token 1 dominant ; token 2 (EOS) légèrement moins bon sauf après 2 tokens générés.
        logits[..., 1] = 2.0
        if seq_len >= 3:
            logits[:, -1, 2] = 3.0
        return logits


def test_decode_batch_greedy_matches_greedy_decode_batch() -> None:
    """``decode_batch`` avec beam 1 délègue au greedy."""
    model = _StubDecodeModel()
    input_values = torch.randn(2, 32)
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    kwargs = {
        "model": model,
        "input_values": input_values,
        "attention_mask": attention_mask,
        "bos_id": 0,
        "eos_id": 2,
        "pad_id": 3,
        "max_new_tokens": 5,
    }
    greedy_out = greedy_decode_batch(**kwargs)
    unified_out = decode_batch(**kwargs, beam_size=1)
    assert torch.equal(greedy_out, unified_out)


def test_beam_decode_batch_runs_and_returns_batch_shape() -> None:
    """Beam search produit un tenseur [batch, seq_len] sans erreur."""
    model = _StubDecodeModel()
    input_values = torch.randn(2, 32)
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    output = beam_decode_batch(
        model=model,
        input_values=input_values,
        attention_mask=attention_mask,
        bos_id=0,
        eos_id=2,
        pad_id=3,
        max_new_tokens=6,
        beam_size=3,
    )
    assert output.shape[0] == 2
    assert output.shape[1] >= 1
    assert (output[:, 0] == 0).all()
