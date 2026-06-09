from __future__ import annotations

import json
from pathlib import Path

import pytest
from Gemini.gemini_common import (
    ENV_GEMINI_API_KEY,
    GeminiRequest,
    MissingGeminiApiKeyError,
    get_gemini_api_key,
)


def test_get_gemini_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_GEMINI_API_KEY, raising=False)
    with pytest.raises(MissingGeminiApiKeyError):
        _ = get_gemini_api_key()


def test_get_gemini_api_key_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_GEMINI_API_KEY, "abc")
    assert get_gemini_api_key() == "abc"


def test_gemini_request_defaults() -> None:
    req = GeminiRequest(model_id="gemini-2.5-flash", prompt="x")
    assert req.temperature == 0.0
    assert req.max_output_tokens == 256
    assert req.thinking_level is None


def test_gemini_request_thinking_level_optional() -> None:
    req = GeminiRequest(
        model_id="gemini-3.5-flash",
        prompt="x",
        thinking_level="minimal",
        max_output_tokens=1024,
    )
    assert req.thinking_level == "minimal"
    assert req.max_output_tokens == 1024


def test_infer_gemini_writes_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Gemini import infer_gemini

    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"RIFFxxxxWAVEfmt ")  # minimal placeholder

    monkeypatch.setenv(ENV_GEMINI_API_KEY, "abc")

    monkeypatch.setattr(infer_gemini, "create_gemini_client", lambda: object())
    monkeypatch.setattr(
        infer_gemini,
        "translate_audio",
        lambda **_: "hello",
    )

    out = tmp_path / "pred.jsonl"
    code = infer_gemini.run_infer_gemini(
        input_audio=audio_path,
        config_path=None,
        model_id="gemini-2.5-flash",
        output=out,
        max_retries=0,
        dry_run=False,
        verbose=False,
    )
    assert code == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["pipeline"] == "gemini_st"
    assert rec["translation"] == "hello"
    assert rec["model_id"] == "gemini-2.5-flash"


def test_evaluate_gemini_writes_eval_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Gemini import evaluate_gemini

    # Fake manifests and audio (placeholders).
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"RIFFxxxxWAVEfmt ")
    manifest_valid = tmp_path / "valid.tsv"
    manifest_test = tmp_path / "test.tsv"
    header = "id\taudio\ttgt_text\n"
    row = f"s1\t{audio_path.as_posix()}\thello world\n"
    manifest_valid.write_text(header + row, encoding="utf-8")
    manifest_test.write_text(header + row, encoding="utf-8")

    # Config points to our tmp manifests and tmp run dir.
    cfg = tmp_path / "cfg.yaml"
    run_dir = tmp_path / "runs" / "fr-en" / "run_001"
    cfg.write_text(
        "\n".join(
            [
                "experiment:",
                '  lang_pair: "fr-en"',
                f'  output_dir: "{run_dir.as_posix()}"',
                "data:",
                f'  valid_manifest: "{manifest_valid.as_posix()}"',
                f'  test_manifest: "{manifest_test.as_posix()}"',
                "model:",
                '  gemini_id: "gemini-2.5-flash"',
                "prompt:",
                '  template: "Translate the French speech to English."',
                "decode:",
                "  temperature: 0.0",
                "  max_output_tokens: 32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv(ENV_GEMINI_API_KEY, "abc")

    monkeypatch.setattr(evaluate_gemini, "create_gemini_client", lambda: object())
    monkeypatch.setattr(
        evaluate_gemini,
        "translate_audio_with_metadata",
        lambda **_: evaluate_gemini.GeminiTranslationResult(
            text="hello world",
            usage=evaluate_gemini.GeminiUsage(),
        ),
    )

    code = evaluate_gemini.run_evaluate_gemini(
        config_path=cfg,
        run_id="run_001",
        output_dir=None,
        limit=1,
        max_retries=0,
        dry_run=False,
        verbose=False,
    )
    assert code == 0

    eval_dir = run_dir / "eval"
    assert (eval_dir / "dev_predictions.txt").is_file()
    assert (eval_dir / "test_predictions.txt").is_file()
    assert (eval_dir / "sacrebleu_dev.txt").is_file()
    assert (eval_dir / "sacrebleu_test.txt").is_file()
    assert (eval_dir / "metrics.json").is_file()
