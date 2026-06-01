#!/usr/bin/env python3
"""
CLI pipeline variante 5 — Pantagruel multimodal (speech_text).

Routeur uniquement : délègue aux modules sous ``5_Pantagruel_multimodal/``.
Cette piste est expérimentale ; la priorité opérationnelle reste `2_speechLLM`.

Usage :
    python 5_Pantagruel_multimodal/pipeline.py train \
      --config 5_Pantagruel_multimodal/configs/fr-en/base.yaml \
      --run-id run_001_multimodal --dry-run
    python 5_Pantagruel_multimodal/pipeline.py evaluate \
      --config 5_Pantagruel_multimodal/configs/fr-en/base.yaml \
      --run-id run_001_multimodal --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTIMODAL_ROOT = PROJECT_ROOT / "5_Pantagruel_multimodal"
STAGES = ("spm", "train", "evaluate", "infer")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attacher les options communes de supervision."""
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)


def _load_stage_module(filename: str, module_label: str):
    """Importer dynamiquement un module de stage de la variante 5."""
    path = MULTIMODAL_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_label, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_spm(args: argparse.Namespace) -> int:
    """Router vers ``3_spm`` via ``multimodal_common``."""
    _MULTIMODAL_ROOT = MULTIMODAL_ROOT
    if str(_MULTIMODAL_ROOT) not in sys.path:
        sys.path.insert(0, str(_MULTIMODAL_ROOT))
    from multimodal_common import run_spm_from_config  # noqa: E402

    if args.dry_run:
        print("[dry-run] spm -> 1_Transformer/3_spm.py (config spm.*)")
        return run_spm_from_config(
            args.config,
            dry_run=True,
            verbose=args.verbose,
            overwrite=args.overwrite,
        )
    return run_spm_from_config(
        args.config,
        dry_run=False,
        verbose=args.verbose,
        overwrite=args.overwrite,
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Router vers `train_multimodal.py`."""
    return _load_stage_module(
        "train_multimodal.py",
        "s3t_multimodal_train",
    ).run_from_namespace(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Router vers `evaluate_multimodal.py`."""
    return _load_stage_module(
        "evaluate_multimodal.py",
        "s3t_multimodal_evaluate",
    ).run_from_namespace(args)


def cmd_infer(args: argparse.Namespace) -> int:
    """Router vers `infer_multimodal.py`."""
    return _load_stage_module(
        "infer_multimodal.py",
        "s3t_multimodal_infer",
    ).run_from_namespace(args)


def _run_stage(
    name: str, handler: Callable[[argparse.Namespace], int], args: argparse.Namespace
) -> int:
    """Afficher une bannière puis exécuter la sous-commande correspondante."""
    if args.dry_run:
        print(f"[dry-run] would run stage: {name}")
        return 0
    print(f"\n{'─' * 60}")
    print(f"Stage: {name.upper()} (5_Pantagruel_multimodal)")
    print(f"{'─' * 60}")
    return handler(args)


def cmd_run(args: argparse.Namespace) -> int:
    """Enchaîner `train -> evaluate -> infer` (ou sous-plage)."""
    from_idx = STAGES.index(args.from_stage)
    to_idx = STAGES.index(args.to_stage)
    if from_idx > to_idx:
        print(
            f"ERROR: --from-stage ({args.from_stage}) must come before "
            f"--to-stage ({args.to_stage})",
            file=sys.stderr,
        )
        return 2

    handlers = {
        "spm": cmd_spm,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
    }
    selected = STAGES[from_idx : to_idx + 1]
    print("=" * 60)
    print("S3T PIPELINE — variante 5_Pantagruel_multimodal")
    print(f"  run-id:   {args.run_id}")
    print(f"  config:   {args.config}")
    print(f"  stages:   {' -> '.join(selected)}")
    print("=" * 60)

    for stage in selected:
        if stage == "spm" and hasattr(args, "overwrite_spm"):
            args.overwrite = args.overwrite_spm
        code = _run_stage(stage, handlers[stage], args)
        if code != 0:
            return code

    print("\n" + "=" * 60)
    print("PIPELINE 5_Pantagruel_multimodal COMPLETE")
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Construire les sous-parseurs puis dispatcher."""
    parser = argparse.ArgumentParser(
        description="Pipeline 5_Pantagruel_multimodal — full speech_text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_spm = subparsers.add_parser(
        "spm",
        help="Entraîner SentencePiece (cibles sentence_like)",
    )
    add_common_args(p_spm)
    p_spm.add_argument(
        "--config",
        type=Path,
        default=MULTIMODAL_ROOT / "configs" / "fr-en" / "base.yaml",
    )
    p_spm.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p_spm.set_defaults(func=cmd_spm)

    p_train = subparsers.add_parser("train", help="Entraîner la variante multimodale")
    add_common_args(p_train)
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--run-id", required=True)
    p_train.add_argument("--output-dir", type=Path, default=None)
    p_train.add_argument("--prefer-cpu", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser(
        "evaluate",
        help="Évaluer la variante multimodale (SacreBLEU)",
    )
    add_common_args(p_eval)
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--checkpoint", type=Path, default=None)
    p_eval.add_argument("--beam-size", type=int, default=5)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.add_argument("--prefer-cpu", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = subparsers.add_parser("infer", help="Inférence sur un WAV")
    add_common_args(p_infer)
    p_infer.add_argument("--checkpoint", type=Path, required=True)
    p_infer.add_argument("--input-audio", type=Path, required=True)
    p_infer.add_argument("--config", type=Path, default=None)
    p_infer.add_argument("--beam-size", type=int, default=5)
    p_infer.add_argument("--prefer-cpu", action="store_true")
    p_infer.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    p_infer.set_defaults(func=cmd_infer)

    p_run = subparsers.add_parser("run", help="Enchaîner train -> evaluate -> infer")
    add_common_args(p_run)
    p_run.add_argument(
        "--config",
        type=Path,
        default=MULTIMODAL_ROOT / "configs" / "fr-en" / "base.yaml",
    )
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--output-dir", type=Path, default=None)
    p_run.add_argument("--beam-size", type=int, default=5)
    p_run.add_argument("--prefer-cpu", action="store_true")
    p_run.add_argument("--from-stage", choices=STAGES, default="spm")
    p_run.add_argument(
        "--overwrite-spm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Remplacer le modèle SPM existant (étape spm uniquement)",
    )
    p_run.add_argument("--to-stage", choices=STAGES, default="evaluate")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
