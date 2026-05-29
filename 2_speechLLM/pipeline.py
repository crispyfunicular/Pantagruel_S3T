#!/usr/bin/env python3
"""
CLI pipeline speechLLM — routeur vers train, evaluate, infer.

Usage :
    python 2_speechLLM/pipeline.py train --config 2_speechLLM/configs/fr-en/b1.yaml --run-id run_001
    python 2_speechLLM/pipeline.py evaluate --config 2_speechLLM/configs/fr-en/b1.yaml --run-id run_001
    python 2_speechLLM/pipeline.py infer --checkpoint runs/.../best.pt --input-audio audio.wav
    python 2_speechLLM/pipeline.py run --config 2_speechLLM/configs/fr-en/b1.yaml --run-id run_001
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_speechllm_package() -> None:
    """Enregistrer le namespace ``speechLLM`` depuis ``2_speechLLM/``."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts_communs.variant_bootstrap import bootstrap_speechllm

    bootstrap_speechllm()


STAGES = ("train", "evaluate", "infer")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Ajouter des arguments communs aux sous-commandes speechLLM."""
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def cmd_train(args: argparse.Namespace) -> int:
    """Router vers `speechLLM.train`."""
    _ensure_speechllm_package()
    from speechLLM.train import run_from_namespace

    return run_from_namespace(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Router vers `speechLLM.evaluate`."""
    _ensure_speechllm_package()
    from speechLLM.evaluate import run_from_namespace

    return run_from_namespace(args)


def cmd_infer(args: argparse.Namespace) -> int:
    """Router vers `speechLLM.infer`."""
    _ensure_speechllm_package()
    from speechLLM.infer import run_from_namespace

    return run_from_namespace(args)


def cmd_run(args: argparse.Namespace) -> int:
    """Enchaîner plusieurs étapes speechLLM (train→evaluate→infer)."""
    from_idx = STAGES.index(args.from_stage)
    to_idx = STAGES.index(args.to_stage)
    if from_idx > to_idx:
        print(
            f"ERROR: --from-stage ({args.from_stage}) must come before --to-stage ({args.to_stage})",
            file=sys.stderr,
        )
        return 2

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "infer": cmd_infer,
    }
    selected = STAGES[from_idx : to_idx + 1]
    print("=" * 60)
    print("SPEECHLLM PIPELINE RUN")
    print(f"  run-id: {args.run_id}")
    print(f"  config: {args.config}")
    print(f"  stages: {' → '.join(selected)}")
    print("=" * 60)

    for stage in selected:
        if args.dry_run:
            print(f"[dry-run] would run stage: {stage}")
            continue
        print(f"\n{'─' * 60}")
        print(f"Stage: {stage.upper()}")
        print(f"{'─' * 60}")
        code = handlers[stage](args)
        if code != 0:
            return code

    print("\n" + "=" * 60)
    print("SPEECHLLM PIPELINE RUN COMPLETE")
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Construire les sous-parseurs et dispatcher vers la sous-commande choisie."""
    parser = argparse.ArgumentParser(
        description="Pipeline speechLLM — Pantagruel + projecteur + LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="Entraîner le projecteur (B1)")
    add_common_args(p_train)
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--run-id", required=True)
    p_train.add_argument("--output-dir", type=Path, default=None)
    p_train.add_argument("--prefer-cpu", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = subparsers.add_parser("evaluate", help="SacreBLEU valid/test")
    add_common_args(p_eval)
    p_eval.add_argument("--config", type=Path, required=True)
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--checkpoint", type=Path, default=None)
    p_eval.add_argument("--beam-size", type=int, default=0)
    p_eval.add_argument("--output-dir", type=Path, default=None)
    p_eval.add_argument("--prefer-cpu", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_infer = subparsers.add_parser("infer", help="Inférence sur un WAV")
    add_common_args(p_infer)
    p_infer.add_argument("--checkpoint", type=Path, required=True)
    p_infer.add_argument("--input-audio", type=Path, required=True)
    p_infer.add_argument("--config", type=Path, default=None)
    p_infer.add_argument("--beam-size", type=int, default=0)
    p_infer.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    p_infer.add_argument("--prefer-cpu", action="store_true")
    p_infer.set_defaults(func=cmd_infer)

    p_run = subparsers.add_parser("run", help="Enchaîner train → evaluate")
    add_common_args(p_run)
    p_run.add_argument("--config", type=Path, required=True)
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--output-dir", type=Path, default=None)
    p_run.add_argument("--prefer-cpu", action="store_true")
    p_run.add_argument("--beam-size", type=int, default=0)
    p_run.add_argument("--from-stage", choices=STAGES, default="train")
    p_run.add_argument("--to-stage", choices=STAGES, default="evaluate")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
