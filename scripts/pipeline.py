#!/usr/bin/env python3
"""
Unified ST pipeline CLI for Pantagruel speech translation replication.

Stages:
    preflight   — Validate machine, GPU, disk, network
    download    — Fetch m-TEDx corpora (OpenSLR-100)
    prepare     — Audio 16 kHz WAV, manifests, text normalization
    spm         — Train SentencePiece tokenizers (train split only)
    train       — Train ST model (encoder + decoder)
    evaluate    — Decode valid/test + SacreBLEU metrics
    infer       — Inference on new audio files
    run         — Chain stages end-to-end

Usage:
    python scripts/pipeline.py preflight
    python scripts/pipeline.py download --langpairs fr-es
    python scripts/pipeline.py prepare --langpair fr-es
    python scripts/pipeline.py spm --langpair fr-es --vocab-size 1000
    python scripts/pipeline.py train --config configs/fr-es/base.yaml --run-id run_001
    python scripts/pipeline.py evaluate --config configs/fr-es/base.yaml --run-id run_001
    python scripts/pipeline.py infer --checkpoint runs/fr-es/run_001/checkpoints/best.pt --input-audio audio.wav
    python scripts/pipeline.py run --langpair fr-es --run-id run_001 --from-stage preflight --to-stage evaluate
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

STAGES = ("preflight", "download", "prepare", "spm", "train", "evaluate", "infer")

EXIT_NOT_IMPLEMENTED = 7


class NotYetImplemented(NotImplementedError):
    """Raised when a pipeline stage is not yet implemented."""


def not_yet(stage: str) -> None:
    raise NotYetImplemented(
        f"NotYetImplemented: stage '{stage}' is not implemented yet."
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without executing.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path.",
    )


def _load_preflight_module():
    path = PROJECT_ROOT / "scripts" / "0_preflight.py"
    spec = importlib.util.spec_from_file_location("s3t_preflight", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load preflight module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_preflight(args: argparse.Namespace) -> int:
    preflight = _load_preflight_module()
    return preflight.run_from_namespace(args)


def _load_download_module():
    path = PROJECT_ROOT / "scripts" / "1_download.py"
    spec = importlib.util.spec_from_file_location("s3t_download", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load download module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_download(args: argparse.Namespace) -> int:
    download = _load_download_module()
    return download.run_from_namespace(args)


def cmd_prepare(args: argparse.Namespace) -> int:
    not_yet("prepare")


def cmd_spm(args: argparse.Namespace) -> int:
    not_yet("spm")


def cmd_train(args: argparse.Namespace) -> int:
    not_yet("train")


def cmd_evaluate(args: argparse.Namespace) -> int:
    not_yet("evaluate")


def cmd_infer(args: argparse.Namespace) -> int:
    not_yet("infer")


def _run_stage(
    name: str, handler: Callable[[argparse.Namespace], int], args: argparse.Namespace
) -> int:
    if args.dry_run:
        print(f"[dry-run] would run stage: {name}")
        return 0
    print(f"\n{'─' * 60}")
    print(f"Stage: {name.upper()}")
    print(f"{'─' * 60}")
    return handler(args)


def cmd_run(args: argparse.Namespace) -> int:
    """Orchestrate pipeline stages from --from-stage to --to-stage."""
    from_idx = STAGES.index(args.from_stage)
    to_idx = STAGES.index(args.to_stage)
    if from_idx > to_idx:
        print(
            f"ERROR: --from-stage ({args.from_stage}) must come before --to-stage ({args.to_stage})",
            file=sys.stderr,
        )
        return 2

    stage_handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "preflight": cmd_preflight,
        "download": cmd_download,
        "prepare": cmd_prepare,
        "spm": cmd_spm,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
    }

    selected = STAGES[from_idx : to_idx + 1]
    print("=" * 60)
    print("S3T PIPELINE RUN")
    print(f"  langpair: {args.langpair}")
    print(f"  run-id:   {args.run_id}")
    print(f"  stages:   {' → '.join(selected)}")
    print("=" * 60)

    for stage in selected:
        try:
            code = _run_stage(stage, stage_handlers[stage], args)
            if code != 0:
                return code
        except NotYetImplemented as exc:
            print(f"  {exc}", file=sys.stderr)
            return EXIT_NOT_IMPLEMENTED

    print("\n" + "=" * 60)
    print("PIPELINE RUN COMPLETE (skeleton — stages may still be placeholders)")
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S3T Pipeline — Pantagruel speech translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- preflight ---
    p_preflight = subparsers.add_parser(
        "preflight", help="Validate environment and resources"
    )
    add_common_args(p_preflight)
    p_preflight.add_argument("--min-disk-gb", type=int, default=200)
    p_preflight.add_argument("--min-vram-gb", type=int, default=8)
    p_preflight.add_argument(
        "--check-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_preflight.add_argument(
        "--check-network",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_preflight.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "preflight_report.json",
    )
    p_preflight.add_argument(
        "--disk-path",
        type=Path,
        default=PROJECT_ROOT,
    )
    p_preflight.add_argument("--network-timeout", type=float, default=10.0)
    p_preflight.set_defaults(func=cmd_preflight)

    # --- download ---
    p_download = subparsers.add_parser(
        "download", help="Download m-TEDx datasets (OpenSLR-100)"
    )
    add_common_args(p_download)
    p_download.add_argument(
        "--langpairs",
        default="fr-en",
        help="Comma-separated language pairs (default: fr-en)",
    )
    p_download.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw",
    )
    p_download.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_download.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p_download.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "download_manifest.json",
    )
    p_download.set_defaults(func=cmd_download)

    # --- prepare ---
    p_prepare = subparsers.add_parser("prepare", help="Prepare audio and manifests")
    add_common_args(p_prepare)
    p_prepare.add_argument("--langpair", required=True, help="e.g. fr-es")
    p_prepare.add_argument(
        "--input-root", type=Path, default=PROJECT_ROOT / "datasets" / "raw"
    )
    p_prepare.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "datasets" / "processed"
    )
    p_prepare.add_argument(
        "--manifests-root", type=Path, default=PROJECT_ROOT / "datasets" / "manifests"
    )
    p_prepare.add_argument("--sample-rate", type=int, default=16000)
    p_prepare.add_argument("--min-duration", type=float, default=1.0)
    p_prepare.add_argument("--max-duration", type=float, default=30.0)
    p_prepare.add_argument("--text-norm", default="nfkc", choices=("nfkc", "none"))
    p_prepare.add_argument("--lowercase", action="store_true", default=False)
    p_prepare.add_argument("--fail-on-leak", action="store_true", default=True)
    p_prepare.set_defaults(func=cmd_prepare)

    # --- spm ---
    p_spm = subparsers.add_parser("spm", help="Train SentencePiece tokenizer")
    add_common_args(p_spm)
    p_spm.add_argument("--langpair", required=True)
    p_spm.add_argument("--vocab-size", type=int, default=1000)
    p_spm.add_argument("--model-type", default="unigram", choices=("unigram", "bpe"))
    p_spm.set_defaults(func=cmd_spm)

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train ST model")
    add_common_args(p_train)
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--run-id", required=True)
    p_train.add_argument("--output-dir", type=Path, default=None)
    p_train.set_defaults(func=cmd_train)

    # --- evaluate ---
    p_evaluate = subparsers.add_parser("evaluate", help="Evaluate with SacreBLEU")
    add_common_args(p_evaluate)
    p_evaluate.add_argument("--config", type=Path, required=True)
    p_evaluate.add_argument("--run-id", required=True)
    p_evaluate.add_argument("--checkpoint", type=Path, default=None)
    p_evaluate.add_argument("--beam-size", type=int, default=5)
    p_evaluate.add_argument("--output-dir", type=Path, default=None)
    p_evaluate.set_defaults(func=cmd_evaluate)

    # --- infer ---
    p_infer = subparsers.add_parser("infer", help="Run inference on audio")
    add_common_args(p_infer)
    p_infer.add_argument("--config", type=Path, default=None)
    p_infer.add_argument("--checkpoint", type=Path, required=True)
    p_infer.add_argument("--input-audio", type=Path, required=True)
    p_infer.add_argument("--beam-size", type=int, default=5)
    p_infer.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    p_infer.set_defaults(func=cmd_infer)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run pipeline stages end-to-end")
    add_common_args(p_run)
    p_run.add_argument("--langpair", default="fr-es")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument(
        "--from-stage",
        default="preflight",
        choices=STAGES,
    )
    p_run.add_argument(
        "--to-stage",
        default="evaluate",
        choices=STAGES,
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return args.func(args) or 0
    except NotYetImplemented as exc:
        print(f"{exc}", file=sys.stderr)
        return EXIT_NOT_IMPLEMENTED


if __name__ == "__main__":
    sys.exit(main())
