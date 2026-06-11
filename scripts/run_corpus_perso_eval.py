#!/usr/bin/env python3
"""
Évaluation BLEU sur le corpus personnel (lectures FR, références EN).

Lit ``corpus_perso/eval_profiles.yaml`` : activer les modèles (``enabled: true``)
et ajuster checkpoints / hyperparamètres de décodage avant de lancer.

Usage :
  python scripts/build_corpus_perso_manifest.py
  python scripts/run_corpus_perso_eval.py --dry-run
  python scripts/run_corpus_perso_eval.py --only gemini_35_v2
  python scripts/run_corpus_perso_eval.py --all -v

Sorties : ``runs/fr-en/eval_corpus_perso_*/eval/metrics.json``
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILES = PROJECT_ROOT / "corpus_perso/eval_profiles.yaml"

PIPELINE_BY_VARIANT: dict[str, tuple[Path, str]] = {
    "transformer": (PROJECT_ROOT / "1_Transformer/pipeline.py", "evaluate"),
    "speechllm": (PROJECT_ROOT / "2_speechLLM/pipeline.py", "evaluate"),
    "gemini": (PROJECT_ROOT / "3_Gemini/pipeline.py", "evaluate"),
    "cascade": (PROJECT_ROOT / "4_cascade/pipeline.py", "evaluate"),
}


def load_profiles(path: Path) -> dict[str, Any]:
    """Charger le fichier YAML des profils d'évaluation."""
    if not path.is_file():
        msg = f"Profils absents: {path}"
        raise FileNotFoundError(msg)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"YAML invalide: {path}"
        raise ValueError(msg)
    return payload


def resolve_models(
    profiles: dict[str, Any],
    *,
    only: list[str] | None,
    run_all: bool,
) -> list[dict[str, Any]]:
    """
    Filtrer les modèles à exécuter selon ``--only`` ou ``enabled``.

    Paramètres :
        profiles : Contenu de ``eval_profiles.yaml``.
        only : Liste d'identifiants à forcer (ignore ``enabled``).
        run_all : Si vrai, exécuter tous les profils même ``enabled: false``.

    Retour :
        Liste de dicts modèle prêts à lancer.
    """
    models = profiles.get("models")
    if not isinstance(models, list):
        msg = "Clé ``models`` manquante dans eval_profiles.yaml"
        raise ValueError(msg)

    selected: list[dict[str, Any]] = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", "")).strip()
        if not model_id:
            continue
        if only is not None:
            if model_id not in only:
                continue
        elif not run_all and not bool(entry.get("enabled", False)):
            continue
        selected.append(entry)
    return selected


def build_eval_command(model: dict[str, Any]) -> list[str]:
    """
    Construire la commande ``pipeline.py evaluate`` pour un profil.

    Paramètres :
        model : Entrée YAML (``pipeline``, ``config``, ``run_id``, etc.).

    Retour :
        Liste d'arguments pour ``subprocess``.

    Lève :
        ValueError : Pipeline ou champs obligatoires manquants.
    """
    pipeline_name = str(model.get("pipeline", "")).strip()
    if pipeline_name not in PIPELINE_BY_VARIANT:
        msg = f"Pipeline inconnu pour {model.get('id')}: {pipeline_name!r}"
        raise ValueError(msg)

    config = model.get("config")
    run_id = model.get("run_id")
    if not config or not run_id:
        msg = f"Profil {model.get('id')}: ``config`` et ``run_id`` requis"
        raise ValueError(msg)

    script_path, subcommand = PIPELINE_BY_VARIANT[pipeline_name]
    cmd = [
        sys.executable,
        str(script_path),
        subcommand,
        "--config",
        str(PROJECT_ROOT / str(config)),
        "--run-id",
        str(run_id),
    ]

    checkpoint = model.get("checkpoint")
    if checkpoint and pipeline_name in {"transformer", "speechllm"}:
        cmd.extend(["--checkpoint", str(PROJECT_ROOT / str(checkpoint))])

    extra_args = model.get("extra_args") or []
    if not isinstance(extra_args, list):
        msg = f"Profil {model.get('id')}: ``extra_args`` doit être une liste"
        raise ValueError(msg)
    cmd.extend(str(arg) for arg in extra_args)
    return cmd


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Évaluer les modèles S3T sur le corpus personnel",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=DEFAULT_PROFILES,
        help=f"YAML des profils (défaut: {DEFAULT_PROFILES})",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Liste d'IDs séparés par des virgules (ex. gemini_35_v2,cascade_utterance)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Lancer tous les profils même si enabled: false",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    try:
        profiles = load_profiles(args.profiles)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest = PROJECT_ROOT / str(
        profiles.get("manifest", "corpus_perso/corpus_perso_test.tsv")
    )
    if not manifest.is_file():
        print(f"ERROR: manifest absent: {manifest}", file=sys.stderr)
        print(
            "  Lancez: python scripts/build_corpus_perso_manifest.py", file=sys.stderr
        )
        return 2

    only = [part.strip() for part in args.only.split(",") if part.strip()] or None
    try:
        models = resolve_models(profiles, only=only, run_all=args.all)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not models:
        print(
            "Aucun modèle sélectionné. Activez ``enabled: true`` dans "
            "corpus_perso/eval_profiles.yaml ou utilisez --only / --all.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print(f"[dry-run] manifest={manifest} — {len(models)} modèle(s)")
        for model in models:
            cmd = build_eval_command(model)
            print(f"  {model['id']}: {' '.join(cmd)}")
        return 0

    exit_code = 0
    for model in models:
        model_id = str(model.get("id"))
        try:
            cmd = build_eval_command(model)
        except ValueError as exc:
            print(f"ERROR [{model_id}]: {exc}", file=sys.stderr)
            exit_code = 2
            continue

        print(f"=== {model_id} ===")
        if args.verbose:
            print(" ".join(cmd))
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            print(
                f"ERROR: {model_id} a échoué (code {result.returncode})",
                file=sys.stderr,
            )
            exit_code = result.returncode

    if exit_code == 0:
        print("Terminé. Scores dans runs/fr-en/eval_corpus_perso_*/eval/metrics.json")
        print("Sync CSV : python scripts_communs/update_experiments_tracking.py --all")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
