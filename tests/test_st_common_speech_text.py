"""Tests encodeur Pantagruel speech_text dans S3TModel."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from scripts_communs.st_common import S3TModel


@patch("scripts_communs.st_common.AutoModel.from_pretrained")
def test_s3t_encode_speech_text_mode(mock_from_pretrained: MagicMock) -> None:
    """``encoder_api=speech_text`` appelle ``mode=AUDIO`` et lit ``audio_output``."""
    mock_encoder = MagicMock()
    mock_encoder.config.hidden_size = 4
    mock_from_pretrained.return_value = mock_encoder

    hidden = torch.randn(2, 5, 4)
    audio_out = MagicMock()
    audio_out.last_hidden_state = hidden
    outputs = MagicMock()
    outputs.audio_output = audio_out
    mock_encoder.return_value = outputs

    model = S3TModel(
        encoder_name="dummy/speech_text",
        vocab_size=32,
        hidden_dim=8,
        decoder_layers=1,
        decoder_heads=2,
        dropout=0.0,
        pad_id=0,
        max_positions=16,
        encoder_api="speech_text",
        trust_remote_code=True,
    )
    model.encoder_proj = torch.nn.Linear(4, 8)

    values = torch.randn(2, 100)
    mask = torch.ones(2, 100, dtype=torch.long)
    memory = model.encode(values, mask)

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.kwargs.get("trust_remote_code") is True
    call_kwargs = mock_encoder.call_args.kwargs
    assert call_kwargs["mode"] == "AUDIO"
    assert memory.shape == (2, 5, 8)
