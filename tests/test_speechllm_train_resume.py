"""Tests reprise et sauvegarde périodique speechLLM (sans Hugging Face)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from speechLLM.speechllm_lib import (
    build_speechllm_checkpoint_payload,
    resolve_speechllm_resume_checkpoint,
)


class _DummyModel(torch.nn.Module):
    """Modèle minimal pour tester la sérialisation checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.projector = torch.nn.Linear(4, 4)


def test_build_checkpoint_payload_includes_train_state(tmp_path: Path) -> None:
    """Le payload doit inclure optimiseur/scaler pour --resume."""
    model = _DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    config = {"model": {"freeze_encoder": True}}

    payload = build_speechllm_checkpoint_payload(
        model=model,
        config=config,
        run_id="run_test",
        git_commit="abc",
        update=42,
        best_bleu_dev=12.5,
        optimizer=optimizer,
        scaler=scaler,
        patience_counter=3,
        start_timestamp_utc="2026-07-06T00:00:00+00:00",
    )

    assert payload["update"] == 42
    assert payload["best_bleu_dev"] == 12.5
    assert payload["patience_counter"] == 3
    assert "projector.weight" in payload["trainable_state"]
    assert "optimizer_state" in payload
    assert payload["start_timestamp_utc"] == "2026-07-06T00:00:00+00:00"


def test_resolve_resume_prefers_last_pt(tmp_path: Path) -> None:
    """La reprise par défaut choisit last.pt avant best.pt."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "best.pt").write_bytes(b"x")
    (ckpt_dir / "last.pt").write_bytes(b"y")

    resolved = resolve_speechllm_resume_checkpoint(ckpt_dir, None)
    assert resolved == ckpt_dir / "last.pt"


def test_resolve_resume_from_explicit(tmp_path: Path) -> None:
    """--resume-from doit pointer vers un fichier existant."""
    ckpt = tmp_path / "custom.pt"
    ckpt.write_bytes(b"z")
    assert resolve_speechllm_resume_checkpoint(tmp_path, ckpt) == ckpt


def test_resolve_resume_from_missing_raises(tmp_path: Path) -> None:
    """Checkpoint explicite absent → FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        resolve_speechllm_resume_checkpoint(tmp_path, tmp_path / "nope.pt")
