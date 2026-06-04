"""Tests pipeline cascade ASR→MT (configs, dry-run, backends)."""

from __future__ import annotations

from pathlib import Path

import Cascade.cascade_common as cascade_common_mod
import Cascade.evaluate_cascade as evaluate_cascade_mod
import pytest
from Cascade.cascade_common import (
    CascadePipelineNotReadyError,
    CascadeSettings,
    _move_batch_to_device,
    cascade_translate_audio,
    clear_cascade_model_cache,
    load_cascade_settings,
    resolve_cascade_config_path,
    transcribe_french,
    translate_french_to_english,
)
from Cascade.evaluate_cascade import run_evaluate_cascade
from Cascade.infer_cascade import run_infer_cascade


def test_resolve_cascade_config_path() -> None:
    path = resolve_cascade_config_path(Path("4_cascade/configs/fr-en/cascade.yaml"))
    assert path.is_file()
    assert path.name == "cascade.yaml"


def test_load_cascade_settings_defaults() -> None:
    settings = load_cascade_settings({})
    assert settings.asr_backend == "whisper"
    assert settings.mt_backend == "marian"
    assert "whisper-large-v3" in settings.asr_model_id
    assert "opus-mt-fr-en" in settings.mt_model_id


def test_move_batch_to_device_preserves_integer_tensors() -> None:
    """Les input_ids Marian doivent rester entiers (pas cast fp16)."""
    import torch

    batch = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }
    moved = _move_batch_to_device(batch, torch.device("cpu"))
    assert moved["input_ids"].dtype == torch.long
    assert moved["attention_mask"].dtype == torch.long


def test_unsupported_asr_backend(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    settings = CascadeSettings(
        asr_backend="pantagruel",
        asr_model_id="x",
        mt_backend="marian",
        mt_model_id="Helsinki-NLP/opus-mt-fr-en",
        asr_language="fr",
        mt_max_length=256,
    )
    with pytest.raises(CascadePipelineNotReadyError, match="ASR backend"):
        transcribe_french(audio, settings)


def test_unsupported_mt_backend() -> None:
    settings = CascadeSettings(
        asr_backend="whisper",
        asr_model_id="openai/whisper-small",
        mt_backend="nllb",
        mt_model_id="y",
        asr_language="fr",
        mt_max_length=256,
    )
    with pytest.raises(CascadePipelineNotReadyError, match="MT backend"):
        translate_french_to_english("bonjour", settings)


def test_translate_empty_french_returns_empty() -> None:
    settings = load_cascade_settings({})
    assert translate_french_to_english("   ", settings) == ""


def test_cascade_translate_audio_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enchaînement ASR→MT sans charger Hugging Face."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    settings = load_cascade_settings({})

    monkeypatch.setattr(
        cascade_common_mod,
        "transcribe_french",
        lambda _path, _settings: "bonjour le monde",
    )
    monkeypatch.setattr(
        cascade_common_mod,
        "translate_french_to_english",
        lambda text_fr, _settings: f"EN:{text_fr}",
    )
    assert cascade_translate_audio(audio, settings) == "EN:bonjour le monde"


def test_evaluate_cascade_dry_run() -> None:
    code = run_evaluate_cascade(
        config_path=Path("4_cascade/configs/fr-en/cascade.yaml"),
        run_id="run_test_cascade_dry",
        output_dir=None,
        limit=0,
        dry_run=True,
        verbose=False,
    )
    assert code == 0


def test_evaluate_cascade_with_mock_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluate complète avec backends simulés (pas de GPU/HF)."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFxxxxWAVEfmt ")
    row = f"s1\t{audio}\thello world\n"
    valid_manifest = tmp_path / "valid.tsv"
    test_manifest = tmp_path / "test.tsv"
    valid_manifest.write_text("id\taudio\ttgt_text\n" + row, encoding="utf-8")
    test_manifest.write_text("id\taudio\ttgt_text\n" + row, encoding="utf-8")

    config_path = tmp_path / "cascade_test.yaml"
    config_path.write_text(
        f"""
experiment:
  name: test
  lang_pair: fr-en
data:
  valid_manifest: "{valid_manifest}"
  test_manifest: "{test_manifest}"
asr:
  backend: whisper
  model_id: openai/whisper-small
mt:
  backend: marian
  model_id: Helsinki-NLP/opus-mt-fr-en
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_cascade_mod,
        "cascade_translate_audio",
        lambda _audio_path, _settings: "hello world",
    )

    code = run_evaluate_cascade(
        config_path=config_path,
        run_id="run_test_cascade_mock",
        output_dir=tmp_path / "run",
        limit=0,
        dry_run=False,
        verbose=False,
    )
    assert code == 0

    eval_dir = tmp_path / "run" / "eval"
    assert (eval_dir / "dev_predictions.txt").read_text(encoding="utf-8").strip() == (
        "hello world"
    )
    assert (eval_dir / "metrics.json").is_file()


def test_infer_cascade_dry_run(tmp_path: Path) -> None:
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    code = run_infer_cascade(
        input_audio=audio,
        config_path=Path("4_cascade/configs/fr-en/cascade.yaml"),
        output=tmp_path / "out.jsonl",
        dry_run=True,
        verbose=False,
    )
    assert code == 0


def teardown_module() -> None:
    """Libérer le cache modèles entre modules de tests."""
    clear_cascade_model_cache()
