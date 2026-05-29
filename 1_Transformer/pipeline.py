#!/usr/bin/env python3
"""
CLI pipeline variante 1 — baseline ST Transformer (étapes 3 à 6).

Routeur uniquement : délègue aux modules sous ``1_Transformer/``.
Prérequis : étapes communes 0–2 via ``scripts_communs/``.

Usage :
    python 1_Transformer/pipeline.py spm --langpair fr-en --vocab-size 1000
    python 1_Transformer/pipeline.py train --config 1_Transformer/configs/fr-en/base.yaml --run-id run_001
    python 1_Transformer/pipeline.py evaluate --config 1_Transformer/configs/fr-en/base.yaml --run-id run_001
    python 1_Transformer/pipeline.py run --langpair fr-en --run-id run_001 --from-stage spm --to-stage evaluate
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSFORMER_ROOT = PROJECT_ROOT / "1_Transformer"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STAGES = ("spm", "train", "evaluate", "infer")
EXIT_NOT_IMPLEMENTED = 7


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Attacher ``--verbose``, ``--dry-run`` et ``--log-file``."""
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", type=Path, default=None)


def _load_stage_module(filename: str, module_label: str):
    """Importer un script numéroté depuis ``1_Transformer/``."""
    path = TRANSFORMER_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_label, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cmd_spm(args: argparse.Namespace) -> int:
    """Exécuter l'entraînement SPM étape 3."""
    return _load_stage_module("3_spm.py", "s3t_spm").run_from_namespace(args)


def cmd_train(args: argparse.Namespace) -> int:
    """Exécuter l'entraînement ST étape 4."""
    return _load_stage_module("4_train.py", "s3t_train").run_from_namespace(args)


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Exécuter l'évaluation SacreBLEU étape 5."""
    return _load_stage_module("5_evaluate.py", "s3t_evaluate").run_from_namespace(args)


def cmd_infer(args: argparse.Namespace) -> int:
    """Exécuter l'inférence étape 6."""
    return _load_stage_module("6_infer.py", "s3t_infer").run_from_namespace(args)


def _run_stage(
    name: str, handler: Callable[[argparse.Namespace], int], args: argparse.Namespace
) -> int:
    """Bannière d'étape + handler (sauf dry-run)."""
    if args.dry_run:
        print(f"[dry-run] would run stage: {name}")
        return 0
    print(f"\n{'─' * 60}")
    print(f"Stage: {name.upper()} (1_Transformer)")
    print(f"{'─' * 60}")
    return handler(args)


def cmd_run(args: argparse.Namespace) -> int:
    """Orchestrer spm → evaluate (ou sous-plage)."""
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
    print("S3T PIPELINE — variante 1_Transformer")
    print(f"  langpair: {args.langpair}")
    print(f"  run-id:   {args.run_id}")
    print(f"  stages:   {' → '.join(selected)}")
    print("=" * 60)

    for stage in selected:
        code = _run_stage(stage, handlers[stage], args)
        if code != 0:
            return code

    print("\n" + "=" * 60)
    print("PIPELINE TRANSFORMER COMPLETE")
    print("=" * 60)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Construire les sous-parseurs et dispatcher."""
    parser = argparse.ArgumentParser(
        description="Pipeline 1_Transformer — baseline ST end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_spm = subparsers.add_parser("spm", help="Entraîner tokenizer SentencePiece")
    add_common_args(p_spm)
    p_spm.add_argument("--langpair", required=True)
    p_spm.add_argument("--vocab-size", type=int, default=1000)
    p_spm.add_argument("--model-type", default="unigram", choices=("unigram", "bpe"))
    p_spm.add_argument(
        "--manifests-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests",
    )
    p_spm.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed" / "spm",
    )
    p_spm.add_argument("--train-text", type=Path, default=None)
    p_spm.add_argument("--character-coverage", type=float, default=1.0)
    p_spm.add_argument(
        "--overwrite", action=argparse.BooleanOptionalAction, default=False
    )
    p_spm.add_argument("--report", type=Path, default=None)
    p_spm.set_defaults(func=cmd_spm)

    p_train = subparsers.add_parser("train", help="Entraîner modèle ST")
    add_common_args(p_train)
    p_train.add_argument("--config", type=Path, required=True)
    p_train.add_argument("--run-id", required=True)
    p_train.add_argument("--output-dir", type=Path, default=None)
    p_train.add_argument("--prefer-cpu", action="store_true", default=False)
    p_train.set_defaults(func=cmd_train)

    p_evaluate = subparsers.add_parser("evaluate", help="Évaluer avec SacreBLEU")
    add_common_args(p_evaluate)
    p_evaluate.add_argument("--config", type=Path, required=True)
    p_evaluate.add_argument("--run-id", required=True)
    p_evaluate.add_argument("--checkpoint", type=Path, default=None)
    p_evaluate.add_argument("--beam-size", type=int, default=5)
    p_evaluate.add_argument("--output-dir", type=Path, default=None)
    p_evaluate.add_argument("--prefer-cpu", action="store_true", default=False)
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_infer = subparsers.add_parser("infer", help="Inférence sur audio")
    add_common_args(p_infer)
    p_infer.add_argument("--config", type=Path, default=None)
    p_infer.add_argument("--checkpoint", type=Path, required=True)
    p_infer.add_argument("--input-audio", type=Path, required=True)
    p_infer.add_argument("--beam-size", type=int, default=5)
    p_infer.add_argument("--prefer-cpu", action="store_true", default=False)
    p_infer.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "inference" / "predictions.jsonl",
    )
    p_infer.set_defaults(func=cmd_infer)

    p_run = subparsers.add_parser("run", help="Enchaîner spm → evaluate")
    add_common_args(p_run)
    p_run.add_argument("--langpair", default="fr-en")
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument(
        "--config",
        type=Path,
        default=TRANSFORMER_ROOT / "configs" / "fr-en" / "base.yaml",
    )
    p_run.add_argument("--from-stage", default="spm", choices=STAGES)
    p_run.add_argument("--to-stage", default="evaluate", choices=STAGES)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
