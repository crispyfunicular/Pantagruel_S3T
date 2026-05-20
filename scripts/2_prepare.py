#!/usr/bin/env python3
"""
Stage 2 — Prepare m-TEDx: segment audio, normalize text, write manifests.

Reads extracted OpenSLR layout under ``<input_root>/mtedx_<langpair>/data/``.
Writes 16 kHz mono PCM16 WAV segments and TSV manifests for train/valid/test.

Usage:
    python scripts/2_prepare.py --langpair fr-en
    python scripts/2_prepare.py --langpair fr-es --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_LANGPAIRS = frozenset({"fr-en", "fr-pt", "fr-es"})
SPLITS = ("train", "valid", "test")

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"

MANIFEST_COLUMNS = (
    "id",
    "audio",
    "n_frames",
    "tgt_text",
    "speaker",
    "tgt_lang",
    "src_text",
    "src_lang",
)


@dataclass
class SegmentRecord:
    utt_id: str
    wav_path: Path
    offset_s: float
    duration_s: float
    src_text: str
    tgt_text: str
    speaker: str
    src_lang: str
    tgt_lang: str


@dataclass
class SplitStats:
    split: str
    segments_in: int = 0
    segments_kept: int = 0
    segments_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)


PROGRESS_INTERVAL = 200


@dataclass
class PrepareReport:
    timestamp_utc: str
    langpair: str
    input_root: str
    output_root: str
    manifests_root: str
    sample_rate: int
    min_duration: float
    max_duration: float
    text_norm: str
    lowercase: bool
    splits: list[SplitStats] = field(default_factory=list)
    leak_issues: list[str] = field(default_factory=list)
    exit_code: int = 0


def parse_langpair(value: str) -> str:
    """Validate a single language pair."""
    pair = value.strip()
    if pair not in SUPPORTED_LANGPAIRS:
        supported = ", ".join(sorted(SUPPORTED_LANGPAIRS))
        raise ValueError(f"Unknown langpair: {pair}. Supported: {supported}")
    return pair


def resolve_corpus_root(input_root: Path, langpair: str) -> Path:
    """Locate extracted m-TEDx tree (OpenSLR uses ``<langpair>/``, not always ``mtedx_*``)."""
    candidates = (
        input_root / f"mtedx_{langpair}",
        input_root / langpair,
    )
    for path in candidates:
        if (path / "data" / "train").is_dir():
            return path
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No m-TEDx corpus for {langpair} under {input_root} (tried: {tried}). "
        "Run stage download first."
    )


def manifest_audio_path(
    output_root: Path,
    langpair: str,
    split: str,
    utt_id: str,
    *,
    repo_root: Path | None = None,
) -> str:
    """POSIX path for manifests: project-relative when under repo, else absolute."""
    abs_audio = (output_root / langpair / split / f"{utt_id}.wav").resolve()
    base = (repo_root or PROJECT_ROOT).resolve()
    try:
        return abs_audio.relative_to(base).as_posix()
    except ValueError:
        return abs_audio.as_posix()


def normalize_text(text: str, *, mode: str, lowercase: bool) -> str:
    """Apply configured text normalization."""
    value = text.strip()
    if mode == "nfkc":
        value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.split())
    if lowercase:
        value = value.lower()
    return value


def iter_mtedx_segments(
    root: Path, langpair: str, split: str
) -> Iterator[SegmentRecord]:
    """Yield segment metadata from m-TEDx YAML + parallel text files."""
    src_lang, tgt_lang = langpair.split("-", maxsplit=1)
    data_dir = root / "data" / split
    wav_root = data_dir / "wav"
    txt_root = data_dir / "txt"
    yaml_path = txt_root / f"{split}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing split descriptor: {yaml_path}")

    with yaml_path.open(encoding="utf-8") as handle:
        segments = yaml.safe_load(handle)
    if not isinstance(segments, list):
        raise ValueError(f"Unexpected YAML content in {yaml_path}")

    src_lines = (
        (txt_root / f"{split}.{src_lang}").read_text(encoding="utf-8").splitlines()
    )
    tgt_lines = (
        (txt_root / f"{split}.{tgt_lang}").read_text(encoding="utf-8").splitlines()
    )
    if len(segments) != len(src_lines) or len(segments) != len(tgt_lines):
        raise ValueError(
            f"Line count mismatch in {split}: yaml={len(segments)}, "
            f"{src_lang}={len(src_lines)}, {tgt_lang}={len(tgt_lines)}"
        )

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, segment in enumerate(segments):
        wav_key = segment["wav"]
        grouped[wav_key].append((index, segment))

    for wav_key in sorted(grouped):
        entries = sorted(grouped[wav_key], key=lambda item: float(item[1]["offset"]))
        flac_name = wav_key.replace(".wav", ".flac")
        wav_path = wav_root / flac_name
        stem = Path(wav_key).stem
        for seg_index, (line_index, segment) in enumerate(entries):
            yield SegmentRecord(
                utt_id=f"{stem}_{seg_index}",
                wav_path=wav_path,
                offset_s=float(segment["offset"]),
                duration_s=float(segment["duration"]),
                src_text=src_lines[line_index].strip(),
                tgt_text=tgt_lines[line_index].strip(),
                speaker=str(segment.get("speaker_id", "")),
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )


def _load_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "soundfile is required for stage prepare. "
            "Install dependencies from requirements.txt."
        ) from exc
    return sf


def _resample_audio(data, src_rate: int, dst_rate: int):
    if src_rate == dst_rate:
        return data
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "torchaudio is required for resampling when source rate != target rate."
        ) from exc
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, src_rate, dst_rate)
    return resampled.squeeze(0).numpy()


def extract_and_save_wav(
    segment: SegmentRecord,
    destination: Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> int:
    """Extract a segment to mono PCM16 WAV; return frame count at target rate."""
    import numpy as np

    sf = _load_soundfile()
    data, sr = sf.read(segment.wav_path.as_posix(), always_2d=True)
    if data.size == 0:
        return 0
    mono = data.mean(axis=1)
    start_frame = int(segment.offset_s * sr)
    end_frame = start_frame + int(segment.duration_s * sr)
    clip = mono[start_frame:end_frame]
    if clip.size == 0:
        return 0
    if sr != sample_rate:
        clip = _resample_audio(clip, sr, sample_rate)
    clip = np.clip(clip.astype("float32"), -1.0, 1.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination.as_posix(), clip, sample_rate, subtype=TARGET_SUBTYPE)
    validation = validate_wav_file(destination, expected_sr=sample_rate)
    if not validation["ok"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"WAV validation failed for {destination.name}: "
            + "; ".join(validation["issues"])
        )
    return int(validation["n_frames"])


def validate_wav_file(
    path: Path,
    *,
    expected_sr: int = TARGET_SAMPLE_RATE,
    expected_frames: int | None = None,
) -> dict[str, Any]:
    """
    Check WAV is consumable by later stages (Pantagruel / SpeechBrain expect 16 kHz mono).

    Validates via soundfile metadata and a torchaudio load smoke-test.
    """
    import numpy as np

    issues: list[str] = []
    sf = _load_soundfile()
    if not path.is_file():
        return {"ok": False, "n_frames": 0, "issues": ["file_missing"]}

    info = sf.info(path.as_posix())
    if info.samplerate != expected_sr:
        issues.append(f"sample_rate={info.samplerate} expected={expected_sr}")
    if info.channels != TARGET_CHANNELS:
        issues.append(f"channels={info.channels} expected={TARGET_CHANNELS}")
    if info.subtype != TARGET_SUBTYPE:
        issues.append(f"subtype={info.subtype} expected={TARGET_SUBTYPE}")

    data, sr = sf.read(path.as_posix(), dtype="float32", always_2d=True)
    n_frames = int(data.shape[0])
    if sr != expected_sr:
        issues.append(f"read_sample_rate={sr}")
    if data.shape[1] != TARGET_CHANNELS:
        issues.append(f"read_channels={data.shape[1]}")
    if not np.isfinite(data).all():
        issues.append("non_finite_samples")
    peak = float(abs(data).max()) if data.size else 0.0
    if peak > 1.0:
        issues.append(f"peak_out_of_range={peak:.4f}")

    if expected_frames is not None and n_frames != expected_frames:
        issues.append(f"n_frames={n_frames} expected={expected_frames}")

    try:
        import torchaudio

        waveform, torch_sr = torchaudio.load(path.as_posix())
        if torch_sr != expected_sr:
            issues.append(f"torchaudio_sr={torch_sr}")
        if waveform.shape[0] != TARGET_CHANNELS:
            issues.append(f"torchaudio_channels={waveform.shape[0]}")
        if waveform.shape[1] != n_frames:
            issues.append(
                f"torchaudio_frames={waveform.shape[1]} soundfile_frames={n_frames}"
            )
    except ImportError:
        pass  # torchaudio smoke-test is optional (e.g. minimal CI env)
    except OSError as exc:
        issues.append(f"torchaudio_load_failed:{exc}")

    return {"ok": not issues, "n_frames": n_frames, "peak": peak, "issues": issues}


def verify_prepared_outputs(
    *,
    langpair: str,
    output_root: Path,
    manifests_root: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_errors: int = 20,
) -> tuple[list[str], dict[str, Any]]:
    """Verify every manifest row points to a valid WAV for train/spm/eval stages."""
    errors: list[str] = []
    summary: dict[str, Any] = {"splits": {}, "total_segments": 0, "invalid_segments": 0}

    for split in SPLITS:
        manifest_path = manifests_root / langpair / f"{split}.tsv"
        if not manifest_path.is_file():
            errors.append(f"Missing manifest: {manifest_path}")
            continue

        split_ok = 0
        split_bad = 0
        with manifest_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                summary["total_segments"] += 1
                audio_rel = row.get("audio", "")
                n_frames_str = row.get("n_frames", "")
                try:
                    expected_frames = int(n_frames_str)
                except ValueError:
                    expected_frames = None

                audio_path = PROJECT_ROOT / audio_rel
                if not audio_path.is_file():
                    split_bad += 1
                    errors.append(f"{split}/{row.get('id')}: missing audio {audio_rel}")
                else:
                    check = validate_wav_file(
                        audio_path,
                        expected_sr=sample_rate,
                        expected_frames=expected_frames,
                    )
                    if not check["ok"]:
                        split_bad += 1
                        errors.append(
                            f"{split}/{row.get('id')}: " + "; ".join(check["issues"])
                        )
                    else:
                        split_ok += 1

                if len(errors) >= max_errors:
                    break

        summary["splits"][split] = {"ok": split_ok, "invalid": split_bad}
        summary["invalid_segments"] += split_bad

    return errors, summary


def write_manifest_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_target_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def detect_leaks(
    *,
    ids_by_split: dict[str, set[str]],
    targets_by_split: dict[str, set[str]],
) -> list[str]:
    """Detect cross-split id or target-text overlap (train leakage)."""
    issues: list[str] = []
    train_ids = ids_by_split.get("train", set())
    train_targets = targets_by_split.get("train", set())
    for split in ("valid", "test"):
        id_overlap = train_ids & ids_by_split.get(split, set())
        if id_overlap:
            issues.append(
                f"ID overlap train/{split}: {len(id_overlap)} utterance id(s)"
            )
        tgt_overlap = train_targets & targets_by_split.get(split, set())
        if tgt_overlap:
            issues.append(
                f"Target text overlap train/{split}: {len(tgt_overlap)} line(s)"
            )
    return issues


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _count_existing_wavs(pair_out: Path) -> int:
    if not pair_out.is_dir():
        return 0
    return sum(1 for _ in pair_out.rglob("*.wav"))


def run_prepare(
    *,
    langpair: str,
    input_root: Path,
    output_root: Path,
    manifests_root: Path,
    sample_rate: int = 16000,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    text_norm: str = "nfkc",
    lowercase: bool = False,
    fail_on_leak: bool = True,
    resume: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    report_path: Path | None = None,
    progress_path: Path | None = None,
) -> tuple[PrepareReport, int]:
    """Prepare one language pair."""
    report_path = report_path or (
        PROJECT_ROOT / "artifacts" / f"prepare_{langpair}.json"
    )
    progress_path = progress_path or (
        PROJECT_ROOT / "artifacts" / f"prepare_{langpair}.progress.json"
    )
    if dry_run:
        try:
            corpus = resolve_corpus_root(input_root, langpair)
        except FileNotFoundError:
            print(f"[dry-run] corpus missing under {input_root}; nothing to prepare")
            report = PrepareReport(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                langpair=langpair,
                input_root=str(input_root.resolve()),
                output_root=str(output_root.resolve()),
                manifests_root=str(manifests_root.resolve()),
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                text_norm=text_norm,
                lowercase=lowercase,
                exit_code=0,
            )
            return report, 0
    else:
        corpus = resolve_corpus_root(input_root, langpair)

    pair_out = output_root / langpair
    pair_manifests = manifests_root / langpair
    if verbose and not dry_run:
        existing = _count_existing_wavs(pair_out)
        print(
            f"Prepare {langpair}: resume={'on' if resume else 'off'}, "
            f"existing_wavs={existing}"
        )
    split_stats: list[SplitStats] = []
    ids_by_split: dict[str, set[str]] = {}
    targets_by_split: dict[str, set[str]] = {}
    processing_errors: list[str] = []

    for split in SPLITS:
        stats = SplitStats(split=split)
        rows: list[dict[str, Any]] = []
        target_lines: list[str] = []
        ids_by_split[split] = set()
        targets_by_split[split] = set()

        if not dry_run:
            split_dir = corpus / "data" / split
            if not split_dir.is_dir():
                processing_errors.append(f"Missing split directory: {split_dir}")
                split_stats.append(stats)
                continue

        try:
            segment_iter = iter_mtedx_segments(corpus, langpair, split)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            processing_errors.append(f"{split}: {exc}")
            split_stats.append(stats)
            continue

        for segment in segment_iter:
            stats.segments_in += 1
            src_text = normalize_text(
                segment.src_text, mode=text_norm, lowercase=lowercase
            )
            tgt_text = normalize_text(
                segment.tgt_text, mode=text_norm, lowercase=lowercase
            )

            if not src_text or not tgt_text:
                stats.segments_dropped += 1
                stats.drop_reasons["empty_text"] = (
                    stats.drop_reasons.get("empty_text", 0) + 1
                )
                continue

            duration = segment.duration_s
            if duration < min_duration:
                stats.segments_dropped += 1
                stats.drop_reasons["too_short"] = (
                    stats.drop_reasons.get("too_short", 0) + 1
                )
                continue
            if duration > max_duration:
                stats.segments_dropped += 1
                stats.drop_reasons["too_long"] = (
                    stats.drop_reasons.get("too_long", 0) + 1
                )
                continue

            if not dry_run and not segment.wav_path.is_file():
                stats.segments_dropped += 1
                stats.drop_reasons["missing_audio"] = (
                    stats.drop_reasons.get("missing_audio", 0) + 1
                )
                continue

            abs_audio = pair_out / split / f"{segment.utt_id}.wav"
            audio_field = manifest_audio_path(
                output_root, langpair, split, segment.utt_id
            )

            if dry_run:
                n_frames = int(duration * sample_rate)
            else:
                reused = False
                if resume and abs_audio.is_file():
                    check = validate_wav_file(abs_audio, expected_sr=sample_rate)
                    if check["ok"]:
                        n_frames = int(check["n_frames"])
                        reused = True
                        stats.drop_reasons["resumed"] = (
                            stats.drop_reasons.get("resumed", 0) + 1
                        )
                    else:
                        abs_audio.unlink(missing_ok=True)
                if not reused:
                    try:
                        n_frames = extract_and_save_wav(
                            segment, abs_audio, sample_rate=sample_rate
                        )
                    except (OSError, RuntimeError, ImportError) as exc:
                        stats.segments_dropped += 1
                        stats.drop_reasons["audio_error"] = (
                            stats.drop_reasons.get("audio_error", 0) + 1
                        )
                        processing_errors.append(f"{split}/{segment.utt_id}: {exc}")
                        continue

            if n_frames <= 0:
                stats.segments_dropped += 1
                stats.drop_reasons["empty_audio"] = (
                    stats.drop_reasons.get("empty_audio", 0) + 1
                )
                if not dry_run and abs_audio.exists():
                    abs_audio.unlink(missing_ok=True)
                continue

            stats.segments_kept += 1
            ids_by_split[split].add(segment.utt_id)
            targets_by_split[split].add(tgt_text)
            rows.append(
                {
                    "id": segment.utt_id,
                    "audio": audio_field,
                    "n_frames": n_frames,
                    "tgt_text": tgt_text,
                    "speaker": segment.speaker,
                    "tgt_lang": segment.tgt_lang,
                    "src_text": src_text,
                    "src_lang": segment.src_lang,
                }
            )
            target_lines.append(tgt_text)

            if not dry_run and stats.segments_in % PROGRESS_INTERVAL == 0:
                _write_progress(
                    progress_path,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "langpair": langpair,
                        "split": split,
                        "resume": resume,
                        "segments_in": stats.segments_in,
                        "segments_kept": stats.segments_kept,
                        "segments_dropped": stats.segments_dropped,
                        "drop_reasons": stats.drop_reasons,
                        "wav_on_disk": _count_existing_wavs(pair_out),
                    },
                )
                if verbose:
                    print(
                        f"    progress {split}: in={stats.segments_in} "
                        f"kept={stats.segments_kept} "
                        f"resumed={stats.drop_reasons.get('resumed', 0)}"
                    )

        if not dry_run:
            write_manifest_tsv(pair_manifests / f"{split}.tsv", rows)
            write_target_lines(pair_manifests / f"{split}.target.txt", target_lines)

        if verbose:
            print(
                f"  {split}: kept {stats.segments_kept}/{stats.segments_in} "
                f"(dropped {stats.segments_dropped})"
            )
        split_stats.append(stats)

    leak_issues = detect_leaks(
        ids_by_split=ids_by_split, targets_by_split=targets_by_split
    )
    wav_errors: list[str] = []
    wav_summary: dict[str, Any] = {}
    if not dry_run and not processing_errors:
        wav_errors, wav_summary = verify_prepared_outputs(
            langpair=langpair,
            output_root=output_root,
            manifests_root=manifests_root,
            sample_rate=sample_rate,
        )
        if verbose and wav_summary:
            print(f"  WAV verification: {wav_summary}")

    exit_code = 0
    if processing_errors:
        exit_code = 4
    if leak_issues and fail_on_leak:
        exit_code = 5 if exit_code == 0 else exit_code
    if wav_errors and exit_code == 0:
        exit_code = 6

    report = PrepareReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        langpair=langpair,
        input_root=str(input_root.resolve()),
        output_root=str(output_root.resolve()),
        manifests_root=str(manifests_root.resolve()),
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        text_norm=text_norm,
        lowercase=lowercase,
        splits=split_stats,
        leak_issues=leak_issues,
        exit_code=exit_code,
    )

    if not dry_run:
        _write_report(
            report_path,
            report,
            processing_errors,
            wav_errors=wav_errors,
            wav_summary=wav_summary,
        )
        if processing_errors:
            print(
                f"\nPrepare FAILED ({len(processing_errors)} error(s)).",
                file=sys.stderr,
            )
            for item in processing_errors[:20]:
                print(f"  - {item}", file=sys.stderr)
        elif leak_issues and fail_on_leak:
            print("\nPrepare FAILED: data leakage detected.", file=sys.stderr)
            for item in leak_issues:
                print(f"  - {item}", file=sys.stderr)
        elif wav_errors:
            print(
                f"\nPrepare FAILED: WAV verification ({len(wav_errors)} issue(s)).",
                file=sys.stderr,
            )
            for item in wav_errors[:20]:
                print(f"  - {item}", file=sys.stderr)
        else:
            print(f"\nPrepare complete. Report: {report_path}")
            print(
                f"  WAV format: {TARGET_SAMPLE_RATE} Hz, mono, {TARGET_SUBTYPE} "
                f"({wav_summary.get('total_segments', 0)} segments checked)"
            )
            if leak_issues:
                print(
                    f"  Warning: {len(leak_issues)} leak check(s) "
                    "(--no-fail-on-leak allowed continuation)",
                    file=sys.stderr,
                )

    return report, exit_code


def _write_report(
    path: Path,
    report: PrepareReport,
    processing_errors: list[str],
    *,
    wav_errors: list[str] | None = None,
    wav_summary: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["processing_errors"] = processing_errors
    payload["wav_validation_errors"] = wav_errors or []
    payload["wav_validation_summary"] = wav_summary or {}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="S3T Stage 2 — Prepare m-TEDx audio and manifests",
    )
    parser.add_argument("--langpair", required=True, help="e.g. fr-en")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed",
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--text-norm", choices=("nfkc", "none"), default="nfkc")
    parser.add_argument("--lowercase", action="store_true", default=False)
    parser.add_argument(
        "--fail-on-leak",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip segments whose output WAV already exists and is valid (default: on)",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Progress JSON path (default: artifacts/prepare_<langpair>.progress.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: artifacts/prepare_<langpair>.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip processing; only verify existing manifests and WAV files",
    )
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    try:
        langpair = parse_langpair(args.langpair)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report_path = args.report or (
        PROJECT_ROOT / "artifacts" / f"prepare_{langpair}.json"
    )

    if args.verify_only:
        wav_errors, wav_summary = verify_prepared_outputs(
            langpair=langpair,
            output_root=args.output_root,
            manifests_root=args.manifests_root,
            sample_rate=args.sample_rate,
        )
        report = PrepareReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            langpair=langpair,
            input_root=str(args.input_root.resolve()),
            output_root=str(args.output_root.resolve()),
            manifests_root=str(args.manifests_root.resolve()),
            sample_rate=args.sample_rate,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            text_norm=args.text_norm,
            lowercase=args.lowercase,
            exit_code=0 if not wav_errors else 6,
        )
        _write_report(
            report_path,
            report,
            [],
            wav_errors=wav_errors,
            wav_summary=wav_summary,
        )
        if wav_errors:
            print(
                f"WAV verification FAILED ({len(wav_errors)} issue(s)).",
                file=sys.stderr,
            )
            for item in wav_errors[:20]:
                print(f"  - {item}", file=sys.stderr)
            return 6
        print(
            f"WAV verification OK — {wav_summary.get('total_segments', 0)} segment(s)."
        )
        print(f"Report: {report_path}")
        return 0

    try:
        _, exit_code = run_prepare(
            langpair=langpair,
            input_root=args.input_root,
            output_root=args.output_root,
            manifests_root=args.manifests_root,
            sample_rate=args.sample_rate,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            text_norm=args.text_norm,
            lowercase=args.lowercase,
            fail_on_leak=args.fail_on_leak,
            resume=args.resume,
            dry_run=args.dry_run,
            verbose=args.verbose,
            report_path=report_path,
            progress_path=args.progress,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
