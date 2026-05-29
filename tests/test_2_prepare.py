"""Tests for scripts_communs/2_prepare.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from conftest import PROJECT_ROOT, load_stage_module

prepare = load_stage_module("2_prepare.py")


def _write_flac(
    path: Path, *, duration_s: float = 2.5, sample_rate: int = 16000
) -> None:
    sf = pytest.importorskip("soundfile")
    import numpy as np

    frames = int(duration_s * sample_rate)
    data = np.zeros(frames, dtype="float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path.as_posix(), data, sample_rate, subtype="PCM_16")


def _write_split(
    root: Path,
    langpair: str,
    split: str,
    *,
    segments: list[dict],
    src_lines: list[str],
    tgt_lines: list[str],
) -> None:
    src_lang, tgt_lang = langpair.split("-")
    txt_dir = root / "data" / split / "txt"
    wav_dir = root / "data" / split / "wav"
    txt_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    (txt_dir / f"{split}.yaml").write_text(
        yaml.dump(segments, allow_unicode=True),
        encoding="utf-8",
    )
    (txt_dir / f"{split}.{src_lang}").write_text(
        "\n".join(src_lines) + "\n",
        encoding="utf-8",
    )
    (txt_dir / f"{split}.{tgt_lang}").write_text(
        "\n".join(tgt_lines) + "\n",
        encoding="utf-8",
    )
    for segment in segments:
        flac = wav_dir / segment["wav"].replace(".wav", ".flac")
        if not flac.exists():
            _write_flac(flac)


def _mini_corpus(tmp_path: Path, langpair: str = "fr-en") -> Path:
    # OpenSLR layout: datasets/raw/fr-en/ (not mtedx_fr-en/)
    root = tmp_path / "raw" / langpair
    segments = [
        {
            "wav": "talk_001.wav",
            "offset": 0.0,
            "duration": 2.5,
            "speaker_id": "spk1",
        },
        {
            "wav": "talk_001.wav",
            "offset": 2.5,
            "duration": 0.5,
        },
    ]
    _write_split(
        root,
        langpair,
        "train",
        segments=segments,
        src_lines=["Bonjour le monde.", "Court."],
        tgt_lines=["Hello world.", "Short."],
    )
    _write_split(
        root,
        langpair,
        "valid",
        segments=[
            {
                "wav": "talk_002.wav",
                "offset": 0.0,
                "duration": 2.0,
                "speaker_id": "spk2",
            }
        ],
        src_lines=["Validation."],
        tgt_lines=["Validation."],
    )
    _write_split(
        root,
        langpair,
        "test",
        segments=[
            {
                "wav": "talk_003.wav",
                "offset": 0.0,
                "duration": 2.0,
                "speaker_id": "spk3",
            }
        ],
        src_lines=["Test phrase."],
        tgt_lines=["Test phrase."],
    )
    return root


def test_parse_langpair_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown langpair"):
        prepare.parse_langpair("fr-xx")


def test_resolve_corpus_root_openslr_layout(tmp_path: Path):
    root = _mini_corpus(tmp_path)
    resolved = prepare.resolve_corpus_root(tmp_path / "raw", "fr-en")
    assert resolved == root


def test_validate_wav_file_accepts_pcm16_mono(tmp_path: Path):
    wav = tmp_path / "source.flac"
    _write_flac(wav, duration_s=1.5)
    segment = prepare.SegmentRecord(
        utt_id="seg0",
        talk_id="source",
        order_idx=0,
        wav_path=wav,
        offset_s=0.0,
        duration_s=1.0,
        src_text="a",
        tgt_text="b",
        speaker="spk",
        src_lang="fr",
        tgt_lang="en",
    )
    out = tmp_path / "out.wav"
    frames = prepare.extract_and_save_wav(segment, out, sample_rate=16000)
    assert frames > 0
    check = prepare.validate_wav_file(out, expected_sr=16000, expected_frames=frames)
    assert check["ok"] is True


def test_normalize_text_nfkc_and_lowercase():
    text = prepare.normalize_text("  Café  ", mode="nfkc", lowercase=True)
    assert text == "café"


def test_dry_run_missing_corpus(tmp_path: Path):
    _, exit_code = prepare.run_prepare(
        langpair="fr-en",
        input_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
        manifests_root=tmp_path / "manifests",
        dry_run=True,
    )
    assert exit_code == 0


def test_run_prepare_filters_short_segments(tmp_path: Path):
    _mini_corpus(tmp_path)
    _, exit_code = prepare.run_prepare(
        langpair="fr-en",
        input_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
        manifests_root=tmp_path / "manifests",
        min_duration=1.0,
        report_path=tmp_path / "report.json",
    )
    assert exit_code == 0

    train_tsv = tmp_path / "manifests" / "fr-en" / "train.tsv"
    with train_tsv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 1
    assert rows[0]["tgt_text"] == "Hello world."
    assert rows[0]["audio"].endswith("talk_001_0.wav")
    assert "fr-en/train" in rows[0]["audio"]


def test_detect_leak_fails(tmp_path: Path):
    root = _mini_corpus(tmp_path)
    # Force identical target in train and valid
    tgt_file = root / "data" / "valid" / "txt" / "valid.en"
    tgt_file.write_text("Hello world.\n", encoding="utf-8")

    _, exit_code = prepare.run_prepare(
        langpair="fr-en",
        input_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
        manifests_root=tmp_path / "manifests",
        fail_on_leak=True,
        report_path=tmp_path / "report.json",
    )
    assert exit_code == 5


def test_dedupe_target_overlap_removes_leak(tmp_path: Path):
    root = _mini_corpus(tmp_path)
    # Force identical target in train and valid; enable dedupe to clean it.
    tgt_file = root / "data" / "valid" / "txt" / "valid.en"
    tgt_file.write_text("Hello world.\n", encoding="utf-8")

    _, exit_code = prepare.run_prepare(
        langpair="fr-en",
        input_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
        manifests_root=tmp_path / "manifests",
        fail_on_leak=True,
        dedupe_target_overlap=True,
        report_path=tmp_path / "report.json",
    )
    assert exit_code == 0

    valid_tgt = (tmp_path / "manifests" / "fr-en" / "valid.target.txt").read_text(
        encoding="utf-8"
    )
    assert "Hello world." not in valid_tgt


def test_build_parser_defaults():
    parser = prepare.build_parser()
    args = parser.parse_args(["--langpair", "fr-en"])
    assert args.langpair == "fr-en"
    assert args.input_root == PROJECT_ROOT / "datasets" / "raw"
    assert args.text_norm == "nfkc"


def test_resume_skips_existing_valid_wav(tmp_path: Path, monkeypatch):
    _mini_corpus(tmp_path)
    output_root = tmp_path / "processed"
    manifests_root = tmp_path / "manifests"

    prepare.run_prepare(
        langpair="fr-en",
        input_root=tmp_path / "raw",
        output_root=output_root,
        manifests_root=manifests_root,
        report_path=tmp_path / "report1.json",
    )
    calls = {"n": 0}
    original = prepare.extract_and_save_wav

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(prepare, "extract_and_save_wav", counted)
    prepare.run_prepare(
        langpair="fr-en",
        input_root=tmp_path / "raw",
        output_root=output_root,
        manifests_root=manifests_root,
        resume=True,
        report_path=tmp_path / "report2.json",
    )
    assert calls["n"] == 0


def test_report_written(tmp_path: Path):
    _mini_corpus(tmp_path)
    report_path = tmp_path / "report.json"
    prepare.run_prepare(
        langpair="fr-en",
        input_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
        manifests_root=tmp_path / "manifests",
        report_path=report_path,
    )
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["langpair"] == "fr-en"
    assert data["exit_code"] == 0
