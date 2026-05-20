#!/usr/bin/env python3
"""
Shared utilities for S3T stages 4/5/6 (train/evaluate/infer).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ManifestSample:
    sample_id: str
    audio_path: Path
    target_text: str


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def deep_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def load_yaml_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return payload


def ensure_project_relative(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def read_manifest(path: Path) -> list[ManifestSample]:
    import csv

    rows: list[ManifestSample] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"id", "audio", "tgt_text"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Missing required columns in manifest: {path}")
        for row in reader:
            rows.append(
                ManifestSample(
                    sample_id=row["id"],
                    audio_path=ensure_project_relative(row["audio"]),
                    target_text=row["tgt_text"].strip(),
                )
            )
    return rows


def load_waveform(path: Path, sample_rate: int) -> torch.Tensor:
    data, sr = sf.read(path.as_posix(), dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise ValueError(
            f"Unexpected sample rate for {path}: {sr} (expected {sample_rate})"
        )
    mono = data.mean(axis=1)
    return torch.from_numpy(mono)


def load_sentencepiece(path: Path) -> spm.SentencePieceProcessor:
    model = spm.SentencePieceProcessor()
    loaded = model.load(path.as_posix())
    if not loaded:
        raise RuntimeError(f"Could not load SentencePiece model: {path}")
    return model


class S3TModel(nn.Module):
    """Minimal ST architecture: HF speech encoder + Transformer decoder."""

    def __init__(
        self,
        *,
        encoder_name: str,
        vocab_size: int,
        hidden_dim: int,
        decoder_layers: int,
        decoder_heads: int,
        dropout: float,
        pad_id: int,
        max_positions: int = 512,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = int(self.encoder.config.hidden_size)
        self.encoder_proj: nn.Module
        if encoder_dim == hidden_dim:
            self.encoder_proj = nn.Identity()
        else:
            self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.tok_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_positions, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=decoder_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def freeze_encoder(self, should_freeze: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not should_freeze

    def encode(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.encoder_proj(encoded)

    def decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError(
                f"Token length {seq_len} exceeds max positions "
                f"{self.pos_embed.num_embeddings}"
            )
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        target = self.tok_embed(tokens) + self.pos_embed(positions)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        target_padding = tokens.eq(self.pad_id)
        decoded = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding,
        )
        return self.output_proj(decoded)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(input_values, attention_mask)
        return self.decode(memory, tokens)


def collate_for_training(
    batch: list[ManifestSample],
    *,
    sp_model: spm.SentencePieceProcessor,
    sample_rate: int,
    max_target_tokens: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> dict[str, torch.Tensor]:
    waves = [load_waveform(sample.audio_path, sample_rate) for sample in batch]
    lengths = [wave.numel() for wave in waves]
    max_wave_len = max(lengths)
    input_values = torch.zeros((len(batch), max_wave_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(batch), max_wave_len), dtype=torch.long)
    for idx, wave in enumerate(waves):
        wave_len = wave.numel()
        input_values[idx, :wave_len] = wave
        attention_mask[idx, :wave_len] = 1

    encoded_targets: list[list[int]] = []
    for sample in batch:
        piece_ids = sp_model.encode(sample.target_text, out_type=int)[
            :max_target_tokens
        ]
        encoded_targets.append(piece_ids)

    target_in: list[list[int]] = [[bos_id, *ids] for ids in encoded_targets]
    target_out: list[list[int]] = [[*ids, eos_id] for ids in encoded_targets]
    max_tgt = max(len(tokens) for tokens in target_in)

    tokens_in = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    tokens_out = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for idx, (tin, tout) in enumerate(zip(target_in, target_out, strict=True)):
        tokens_in[idx, : len(tin)] = torch.tensor(tin, dtype=torch.long)
        tokens_out[idx, : len(tout)] = torch.tensor(tout, dtype=torch.long)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def greedy_decode_batch(
    *,
    model: S3TModel,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(input_values, attention_mask)
        batch_size = input_values.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=bos_id,
            dtype=torch.long,
            device=input_values.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_values.device)
        for _ in range(max_new_tokens):
            logits = model.decode(memory, generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, pad_id), next_token
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)
            if bool(torch.all(finished)):
                break
    return generated


def decode_ids_to_text(
    token_ids: list[int],
    *,
    sp_model: spm.SentencePieceProcessor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> str:
    kept: list[int] = []
    for token in token_ids:
        if token in (bos_id, pad_id):
            continue
        if token == eos_id:
            break
        kept.append(token)
    if not kept:
        return ""
    return sp_model.decode(kept)


def resolve_run_dir(
    config: dict[str, Any],
    *,
    run_id: str,
    output_dir_override: Path | None,
) -> Path:
    if output_dir_override is not None:
        return output_dir_override
    from_config = deep_get(config, "experiment.output_dir", None)
    if from_config:
        base = ensure_project_relative(from_config)
        return base.parent / run_id if base.name != run_id else base
    langpair = str(deep_get(config, "experiment.lang_pair", "fr-en"))
    return PROJECT_ROOT / "runs" / langpair / run_id


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
