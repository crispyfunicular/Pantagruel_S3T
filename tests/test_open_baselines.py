"""Tests pipeline open baselines ST (configs, dry-run, backends mockés)."""

from __future__ import annotations

from pathlib import Path

import OpenBaselines.evaluate_open as evaluate_open_mod
import OpenBaselines.open_common as open_common_mod
import pytest
from OpenBaselines.evaluate_open import run_evaluate_open
from OpenBaselines.infer_open import run_infer_open
from OpenBaselines.open_common import (
    OpenModelSettings,
    OpenPipelineNotReadyError,
    _canary_lang_code,
    _seamless_text_token_ids,
    clear_open_model_cache,
    load_open_settings,
    open_translate_audio,
    resolve_open_config_path,
)


def test_resolve_open_config_path() -> None:
    path = resolve_open_config_path(
        Path("6_open_baselines/configs/fr-en/whisper_large_v3_st.yaml")
    )
    assert path.is_file()
    assert path.name == "whisper_large_v3_st.yaml"


def test_load_open_settings_whisper_defaults() -> None:
    settings = load_open_settings(
        {
            "model": {
                "type": "whisper_st",
            }
        }
    )
    assert settings.model_type == "whisper_st"
    assert "whisper-large-v3" in settings.model_id
    assert settings.whisper_language == "fr"


def test_load_open_settings_seamless() -> None:
    settings = load_open_settings({"model": {"type": "seamless_m4t_v2"}})
    assert settings.model_type == "seamless_m4t_v2"
    assert "seamless-m4t-v2" in settings.model_id
    assert settings.src_lang == "fra"
    assert settings.tgt_lang == "eng"


def test_canary_lang_code_maps_iso639_3() -> None:
    assert _canary_lang_code("fra") == "fr"
    assert _canary_lang_code("eng") == "en"


def test_seamless_text_token_ids_tensor() -> None:
    import torch

    generated = torch.tensor([[1, 2, 3]])
    assert _seamless_text_token_ids(generated) == [1, 2, 3]


def test_seamless_text_token_ids_model_output() -> None:
    class FakeOutput:
        sequences = [[10, 11, 12]]

    assert _seamless_text_token_ids(FakeOutput()) == [10, 11, 12]


def test_canary_transcribe_api_on_engine() -> None:
    """_Canary1bEngine utilise l'API NeMo 2.x (audio=, task=ast)."""
    captured: dict = {}

    class FakeModel:
        def transcribe(self, **kwargs):
            captured.update(kwargs)
            return [type("Out", (), {"text": "translated"})()]

    engine = open_common_mod._Canary1bEngine.__new__(open_common_mod._Canary1bEngine)
    engine.model = FakeModel()
    out = engine.translate_file(Path("/tmp/x.wav"), "fra", "eng")
    assert out == "translated"
    assert captured["task"] == "ast"
    assert captured["source_lang"] == "fr"
    assert captured["target_lang"] == "en"
    assert captured["audio"] == [str(Path("/tmp/x.wav").resolve())]


def test_canary_translate_uses_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """open_translate_audio délègue au moteur Canary."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")

    class FakeEngine:
        def translate_file(self, audio_path: Path, src_lang: str, tgt_lang: str) -> str:
            assert src_lang == "fra"
            assert tgt_lang == "eng"
            return "hello"

    clear_open_model_cache()
    monkeypatch.setattr(
        open_common_mod,
        "_get_canary_engine",
        lambda _settings: FakeEngine(),
    )
    settings = load_open_settings({"model": {"type": "canary_1b"}})
    assert open_translate_audio(audio, settings) == "hello"


def test_unsupported_model_type(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    settings = OpenModelSettings(
        model_type="unknown_model",
        model_id="x",
        src_lang="fra",
        tgt_lang="eng",
        whisper_language="fr",
        device="cpu",
        dtype="float32",
        max_new_tokens=256,
    )
    with pytest.raises(OpenPipelineNotReadyError, match="non supporté"):
        open_translate_audio(audio, settings)


def test_open_translate_audio_monkeypatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Traduction simulée sans charger Hugging Face."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    settings = load_open_settings({"model": {"type": "whisper_st"}})

    monkeypatch.setattr(
        open_common_mod,
        "_get_whisper_st_engine",
        lambda _settings: type(
            "Engine",
            (),
            {
                "translate_file": lambda _self, _path, _sr: "hello world",
            },
        )(),
    )
    assert open_translate_audio(audio, settings) == "hello world"


def test_evaluate_open_dry_run() -> None:
    code = run_evaluate_open(
        config_path=Path("6_open_baselines/configs/fr-en/whisper_large_v3_st.yaml"),
        run_id="run_test_open_dry",
        output_dir=None,
        limit=0,
        dry_run=True,
        verbose=False,
    )
    assert code == 0


def test_evaluate_open_with_mock_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evaluate complète avec backend simulé (pas de GPU/HF)."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFFxxxxWAVEfmt ")
    row = f"s1\t{audio}\thello world\n"
    valid_manifest = tmp_path / "valid.tsv"
    test_manifest = tmp_path / "test.tsv"
    valid_manifest.write_text("id\taudio\ttgt_text\n" + row, encoding="utf-8")
    test_manifest.write_text("id\taudio\ttgt_text\n" + row, encoding="utf-8")

    config_path = tmp_path / "open_test.yaml"
    config_path.write_text(
        f"""
experiment:
  name: test
  lang_pair: fr-en
data:
  valid_manifest: "{valid_manifest}"
  test_manifest: "{test_manifest}"
model:
  type: whisper_st
  model_id: openai/whisper-small
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluate_open_mod,
        "open_translate_audio",
        lambda _audio_path, _settings: "hello world",
    )

    code = run_evaluate_open(
        config_path=config_path,
        run_id="run_test_open_mock",
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
    assert (eval_dir / "protocol.json").is_file()


def test_infer_open_dry_run(tmp_path: Path) -> None:
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    code = run_infer_open(
        input_audio=audio,
        config_path=Path("6_open_baselines/configs/fr-en/whisper_large_v3_st.yaml"),
        output=tmp_path / "out.jsonl",
        dry_run=True,
        verbose=False,
    )
    assert code == 0


def teardown_module() -> None:
    """Libérer le cache modèles entre modules de tests."""
    clear_open_model_cache()
