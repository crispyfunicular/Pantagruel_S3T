from __future__ import annotations

from pathlib import Path

import pytest
from Cascade.cascade_common import (
    CascadePipelineNotReadyError,
    CascadeSettings,
    cascade_translate_audio,
    load_cascade_settings,
    resolve_cascade_config_path,
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


def test_cascade_translate_audio_not_ready(tmp_path: Path) -> None:
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    settings = CascadeSettings(
        asr_backend="whisper",
        asr_model_id="x",
        mt_backend="marian",
        mt_model_id="y",
        asr_language="fr",
        mt_max_length=256,
    )
    with pytest.raises(CascadePipelineNotReadyError):
        cascade_translate_audio(audio, settings)


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


def test_evaluate_cascade_not_implemented_when_manifests_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sans dry-run, retourne 3 tant que ASR/MT ne sont pas câblés."""
    from Cascade import evaluate_cascade

    valid = Path("datasets/manifests/fr-en/valid.tsv")
    if not valid.is_file():
        pytest.skip("utterance manifests not prepared")

    monkeypatch.setattr(
        evaluate_cascade,
        "read_manifest",
        lambda _path: [
            type(
                "S",
                (),
                {
                    "sample_id": "x",
                    "audio_path": valid,
                    "target_text": "hi",
                },
            )()
        ],
    )
    code = run_evaluate_cascade(
        config_path=Path("4_cascade/configs/fr-en/cascade.yaml"),
        run_id="run_test_cascade_stub",
        output_dir=None,
        limit=1,
        dry_run=False,
        verbose=False,
    )
    assert code == 3


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
