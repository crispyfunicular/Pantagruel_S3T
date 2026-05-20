#!/usr/bin/env python3
"""Quick FR->FR ASR evaluation with Hugging Face backends.

`PantagrueLLM/Speech_Text_Base_fr_1K_4GB` is a pretraining encoder (no text decoder).
Use `--transcription pantagruel-encoder` to validate loading and forward pass, and
`--transcription whisper` for end-to-end WER/CER until a fairseq ST/ASR checkpoint
is available.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils_text_eval import (  # noqa: E402
    char_error_rate,
    normalize_text_for_eval,
    word_error_rate,
)

TARGET_SAMPLE_RATE = 16_000
DEFAULT_MODEL_ID = "PantagrueLLM/Speech_Text_Base_fr_1K_4GB"
DEFAULT_WHISPER_ID = "openai/whisper-small"
TranscriptionBackend = Literal["pantagruel-encoder", "whisper"]


@dataclass
class SampleResult:
    utt_id: str
    audio_path: str
    reference_path: str
    reference_raw: str
    reference_norm: str
    hypothesis_raw: str
    hypothesis_norm: str
    wer: float | None = None
    cer: float | None = None
    encoder_frames: int | None = None
    encoder_dim: int | None = None
    error: str | None = None


@dataclass
class EvalReport:
    created_at: str
    model_id: str
    whisper_model_id: str | None
    transcription_backend: str
    corpus_dir: str
    text_norm: str
    lowercase: bool
    device: str
    n_evaluated: int = 0
    n_skipped: int = 0
    samples: list[SampleResult] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _load_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            "soundfile is required. Install runtime deps: pip install -r requirements.txt"
        ) from exc
    return sf


def _load_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise SystemExit(
            "torchaudio is required for resampling. Install PyTorch audio extras."
        ) from exc
    return torchaudio


def load_mono_audio(path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> torch.Tensor:
    """Load WAV/FLAC as mono float tensor [1, T] at target sample rate."""
    sf = _load_soundfile()
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        torchaudio = _load_torchaudio()
        tensor = torch.from_numpy(audio).unsqueeze(0)
        audio = (
            torchaudio.functional.resample(tensor, sr, sample_rate).squeeze(0).numpy()
        )
    return torch.from_numpy(audio).unsqueeze(0)


def discover_pairs(
    corpus_dir: Path,
) -> tuple[list[tuple[str, Path, Path]], list[dict[str, str]]]:
    """Pair *.wav with *.lab by basename; report incomplete entries."""
    wavs = {p.stem: p for p in corpus_dir.glob("*.wav")}
    labs = {p.stem: p for p in corpus_dir.glob("*.lab")}
    pairs: list[tuple[str, Path, Path]] = []
    skipped: list[dict[str, str]] = []

    for stem in sorted(wavs.keys() & labs.keys()):
        pairs.append((stem, wavs[stem], labs[stem]))

    for stem in sorted(wavs.keys() - labs.keys()):
        skipped.append(
            {
                "utt_id": stem,
                "reason": "audio_without_transcript",
                "audio": str(wavs[stem]),
            }
        )
    for stem in sorted(labs.keys() - wavs.keys()):
        skipped.append(
            {
                "utt_id": stem,
                "reason": "transcript_without_audio",
                "reference": str(labs[stem]),
            }
        )
    return pairs, skipped


def read_reference(lab_path: Path) -> str:
    return lab_path.read_text(encoding="utf-8").strip()


class PantagruelEncoderBackend:
    """Forward pass on Pantagruel HF encoder (no transcription head)."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        from transformers import AutoModel

        self.model_id = model_id
        self.device = device
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

    def transcribe(self, audio: torch.Tensor) -> tuple[str, dict[str, int]]:
        audio = audio.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_values=audio, mode="AUDIO")
        hidden = outputs.audio_output.last_hidden_state
        meta = {
            "encoder_frames": int(hidden.shape[1]),
            "encoder_dim": int(hidden.shape[2]),
        }
        return "", meta


class WhisperBackend:
    """French ASR proxy via Whisper (for protocol testing, not Pantagruel ST)."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.model_id = model_id
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        forced_ids = self.processor.get_decoder_prompt_ids(
            language="fr", task="transcribe"
        )
        self.forced_decoder_ids = forced_ids

    def transcribe(self, audio: torch.Tensor) -> tuple[str, dict[str, int]]:
        inputs = self.processor(
            audio.squeeze(0).numpy(),
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
            )
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text, {}


def build_backend(
    name: TranscriptionBackend,
    *,
    pantagruel_model_id: str,
    whisper_model_id: str,
    device: torch.device,
) -> PantagruelEncoderBackend | WhisperBackend:
    if name == "pantagruel-encoder":
        return PantagruelEncoderBackend(pantagruel_model_id, device)
    if name == "whisper":
        return WhisperBackend(whisper_model_id, device)
    raise ValueError(f"unsupported backend: {name}")


def evaluate_corpus(
    corpus_dir: Path,
    *,
    model_id: str,
    transcription: TranscriptionBackend,
    whisper_model_id: str,
    text_norm: str,
    lowercase: bool,
    device: torch.device,
    limit: int | None,
) -> EvalReport:
    pairs, skipped = discover_pairs(corpus_dir)
    if limit is not None:
        pairs = pairs[:limit]

    backend = build_backend(
        transcription,
        pantagruel_model_id=model_id,
        whisper_model_id=whisper_model_id,
        device=device,
    )

    report = EvalReport(
        created_at=datetime.now(UTC).isoformat(),
        model_id=model_id,
        whisper_model_id=whisper_model_id if transcription == "whisper" else None,
        transcription_backend=transcription,
        corpus_dir=str(corpus_dir.resolve()),
        text_norm=text_norm,
        lowercase=lowercase,
        device=str(device),
        skipped=skipped,
    )

    if transcription == "pantagruel-encoder":
        report.notes.append(
            "Pantagruel Speech_Text_Base_fr_1K_4GB is a pretraining encoder only; "
            "WER/CER require --transcription whisper or a fairseq fine-tuned checkpoint."
        )

    wers: list[float] = []
    cers: list[float] = []

    for utt_id, wav_path, lab_path in pairs:
        ref_raw = read_reference(lab_path)
        ref_norm = normalize_text_for_eval(ref_raw, mode=text_norm, lowercase=lowercase)
        sample = SampleResult(
            utt_id=utt_id,
            audio_path=str(wav_path.resolve()),
            reference_path=str(lab_path.resolve()),
            reference_raw=ref_raw,
            reference_norm=ref_norm,
            hypothesis_raw="",
            hypothesis_norm="",
        )
        try:
            audio = load_mono_audio(wav_path)
            hyp_raw, meta = backend.transcribe(audio)
            sample.hypothesis_raw = hyp_raw
            sample.hypothesis_norm = normalize_text_for_eval(
                hyp_raw, mode=text_norm, lowercase=lowercase
            )
            sample.encoder_frames = meta.get("encoder_frames")
            sample.encoder_dim = meta.get("encoder_dim")

            if transcription == "whisper" and sample.hypothesis_norm:
                wer = word_error_rate(ref_norm, sample.hypothesis_norm)
                cer = char_error_rate(ref_norm, sample.hypothesis_norm)
                sample.wer = wer.rate
                sample.cer = cer.rate
                wers.append(wer.rate)
                cers.append(cer.rate)
            report.samples.append(sample)
            report.n_evaluated += 1
        except Exception as exc:  # noqa: BLE001 — collect per-utt failures
            sample.error = str(exc)
            report.samples.append(sample)
            report.n_skipped += 1

    report.n_skipped += len(skipped)
    if wers:
        report.aggregate = {
            "wer_mean": statistics.mean(wers),
            "wer_median": statistics.median(wers),
            "cer_mean": statistics.mean(cers),
            "cer_median": statistics.median(cers),
            "n_with_metrics": len(wers),
        }
    elif transcription == "pantagruel-encoder":
        frames = [s.encoder_frames for s in report.samples if s.encoder_frames]
        report.aggregate = {
            "encoder_frames_mean": statistics.mean(frames) if frames else None,
            "n_encoder_forward_ok": len(frames),
        }

    return report


def evaluate_single(
    audio_path: Path,
    *,
    reference: str | None,
    model_id: str,
    transcription: TranscriptionBackend,
    whisper_model_id: str,
    text_norm: str,
    lowercase: bool,
    device: torch.device,
) -> EvalReport:
    backend = build_backend(
        transcription,
        pantagruel_model_id=model_id,
        whisper_model_id=whisper_model_id,
        device=device,
    )
    report = EvalReport(
        created_at=datetime.now(UTC).isoformat(),
        model_id=model_id,
        whisper_model_id=whisper_model_id if transcription == "whisper" else None,
        transcription_backend=transcription,
        corpus_dir=str(audio_path.parent.resolve()),
        text_norm=text_norm,
        lowercase=lowercase,
        device=str(device),
    )
    if transcription == "pantagruel-encoder":
        report.notes.append(
            "Encoder-only mode: hypothesis is empty; use --transcription whisper for text."
        )

    ref_raw = reference or ""
    ref_norm = (
        normalize_text_for_eval(ref_raw, mode=text_norm, lowercase=lowercase)
        if ref_raw
        else ""
    )
    sample = SampleResult(
        utt_id=audio_path.stem,
        audio_path=str(audio_path.resolve()),
        reference_path="",
        reference_raw=ref_raw,
        reference_norm=ref_norm,
        hypothesis_raw="",
        hypothesis_norm="",
    )
    audio = load_mono_audio(audio_path)
    hyp_raw, meta = backend.transcribe(audio)
    sample.hypothesis_raw = hyp_raw
    sample.hypothesis_norm = normalize_text_for_eval(
        hyp_raw, mode=text_norm, lowercase=lowercase
    )
    sample.encoder_frames = meta.get("encoder_frames")
    sample.encoder_dim = meta.get("encoder_dim")
    if ref_norm and sample.hypothesis_norm:
        sample.wer = word_error_rate(ref_norm, sample.hypothesis_norm).rate
        sample.cer = char_error_rate(ref_norm, sample.hypothesis_norm).rate
        report.aggregate = {"wer": sample.wer, "cer": sample.cer}
    report.samples = [sample]
    report.n_evaluated = 1
    return report


def write_report(report: EvalReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **asdict(report),
        "samples": [asdict(s) for s in report.samples],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quick FR->FR ASR evaluation (Pantagruel encoder or Whisper proxy)."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Audio file (.wav) or directory with .wav + .lab pairs",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Pantagruel HF model id (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--transcription",
        choices=["pantagruel-encoder", "whisper"],
        default="whisper",
        help="pantagruel-encoder: forward pass only; whisper: ASR proxy for WER/CER",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_ID,
        help=f"Whisper model when --transcription whisper (default: {DEFAULT_WHISPER_ID})",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference transcript for single-file mode (optional)",
    )
    parser.add_argument("--text-norm", choices=["none", "nfkc"], default="nfkc")
    parser.add_argument("--lowercase", action="store_true", default=True)
    parser.add_argument("--no-lowercase", action="store_false", dest="lowercase")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: artifacts/quick_eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max utterances (batch)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_path: Path = args.input_path
    device = torch.device(args.device)

    if not input_path.exists():
        parser.error(f"path not found: {input_path}")

    t0 = time.perf_counter()
    if input_path.is_file():
        report = evaluate_single(
            input_path,
            reference=args.reference,
            model_id=args.model,
            transcription=args.transcription,
            whisper_model_id=args.whisper_model,
            text_norm=args.text_norm,
            lowercase=args.lowercase,
            device=device,
        )
    elif input_path.is_dir():
        report = evaluate_corpus(
            input_path,
            model_id=args.model,
            transcription=args.transcription,
            whisper_model_id=args.whisper_model,
            text_norm=args.text_norm,
            lowercase=args.lowercase,
            device=device,
            limit=args.limit,
        )
    else:
        parser.error(f"not a file or directory: {input_path}")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        PROJECT_ROOT / "artifacts" / f"quick_eval_{input_path.stem}_{ts}.json"
    )
    write_report(report, output)

    elapsed = time.perf_counter() - t0
    print(f"report: {output}")
    print(f"backend: {report.transcription_backend}")
    print(f"evaluated: {report.n_evaluated}, skipped/incomplete: {report.n_skipped}")
    if report.aggregate:
        print(f"aggregate: {report.aggregate}")
    for note in report.notes:
        print(f"note: {note}")
    print(f"elapsed_s: {elapsed:.1f}")

    if report.transcription_backend == "pantagruel-encoder":
        return 0
    if report.n_evaluated == 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
