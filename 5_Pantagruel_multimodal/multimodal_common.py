#!/usr/bin/env python3
"""
Utilitaires partagés — variante 5 Pantagruel multimodal (speech_text).

Délègue les étapes lourdes au pipeline ``1_Transformer`` (SPM, train, evaluate, infer)
avec une config YAML dédiée (encodeur ``Speech_Text_*``, manifests ``sentence_like``).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from scripts_communs.st_common import (
    PROJECT_ROOT,
    deep_get,
    load_yaml_config,
    resolve_run_dir,
    write_json,
)

TRANSFORMER_ROOT = PROJECT_ROOT / "1_Transformer"
PIPELINE_NAME = "pantagruel_multimodal"


def _load_transformer_module(filename: str, module_label: str):
    """Importer un script numéroté sous ``1_Transformer/``."""
    path = TRANSFORMER_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_label, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_config_path(config_path: Path) -> Path:
    """Résoudre un chemin de config relatif à la racine du dépôt."""
    path = config_path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def ensure_train_target_text(
    train_text_path: Path,
    *,
    train_manifest: Path | None = None,
    verbose: bool = False,
) -> int:
    """
    Créer ``train.target.txt`` depuis ``train.tsv`` si absent (prepare non rejoué).

    Utile sur machine distante lorsque les manifests TSV existent mais pas les
    fichiers cibles ligne-à-ligne attendus par ``3_spm``.

    Retour :
        0 si le fichier existe ou a été généré, 2 si manifest introuvable ou vide.
    """
    if train_text_path.is_file():
        return 0

    manifest_path = train_manifest or (train_text_path.parent / "train.tsv")
    if not manifest_path.is_file():
        print(
            f"ERROR: missing train manifest for SPM: {manifest_path}",
            file=sys.stderr,
        )
        return 2

    train_text_path.parent.mkdir(parents=True, exist_ok=True)
    line_count = 0
    with (
        manifest_path.open(encoding="utf-8") as handle_in,
        train_text_path.open("w", encoding="utf-8") as handle_out,
    ):
        reader = csv.DictReader(handle_in, delimiter="\t")
        if not reader.fieldnames or "tgt_text" not in reader.fieldnames:
            print(
                f"ERROR: manifest missing tgt_text column: {manifest_path}",
                file=sys.stderr,
            )
            return 2
        for row in reader:
            text = str(row.get("tgt_text", "")).strip()
            if not text:
                continue
            handle_out.write(text + "\n")
            line_count += 1

    if line_count == 0:
        print(f"ERROR: no tgt_text lines in {manifest_path}", file=sys.stderr)
        train_text_path.unlink(missing_ok=True)
        return 2

    if verbose:
        print(f"Generated {train_text_path} ({line_count} lines) from {manifest_path}")
    return 0


def run_spm_from_config(
    config_path: Path,
    *,
    dry_run: bool,
    verbose: bool,
    overwrite: bool = False,
) -> int:
    """
    Entraîner SentencePiece selon la section ``spm`` de la config variante 5.

    Retour :
        Code de sortie de ``3_spm.run_spm`` (0 succès).
    """
    config = load_yaml_config(resolve_config_path(config_path))
    langpair = str(deep_get(config, "spm.langpair", "fr-en"))
    vocab_size = int(deep_get(config, "spm.vocab_size", 1000))
    model_type = str(deep_get(config, "spm.model_type", "unigram"))
    manifests_root = PROJECT_ROOT / str(
        deep_get(config, "spm.manifests_root", "datasets/manifests_sentence")
    )
    output_dir = PROJECT_ROOT / str(
        deep_get(config, "spm.output_dir", "datasets/processed/spm")
    )
    train_text = deep_get(config, "spm.train_text", None)
    train_text_path = (
        PROJECT_ROOT / str(train_text)
        if train_text is not None
        else manifests_root / langpair / "train.target.txt"
    )
    train_manifest = PROJECT_ROOT / str(
        deep_get(
            config,
            "data.train_manifest",
            f"datasets/manifests_sentence/{langpair}/train.tsv",
        )
    )
    character_coverage = float(deep_get(config, "spm.character_coverage", 1.0))
    report = deep_get(config, "spm.report", None)
    report_path = PROJECT_ROOT / str(report) if report is not None else None

    code = ensure_train_target_text(
        train_text_path,
        train_manifest=train_manifest,
        verbose=verbose,
    )
    if code != 0:
        return code

    spm_module = _load_transformer_module("3_spm.py", "s3t_multimodal_spm")
    return spm_module.run_spm(
        langpair=langpair,
        vocab_size=vocab_size,
        model_type=model_type,
        manifests_root=manifests_root,
        output_dir=output_dir,
        train_text=train_text_path,
        character_coverage=character_coverage,
        overwrite=overwrite,
        dry_run=dry_run,
        verbose=verbose,
        report_path=report_path,
    )


def delegate_transformer_train(
    *,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """Déléguer l'entraînement ST à ``1_Transformer/4_train.py``."""
    train_module = _load_transformer_module("4_train.py", "s3t_multimodal_train")
    return train_module.run_train(
        config_path=resolve_config_path(config_path),
        run_id=run_id,
        output_dir=output_dir,
        dry_run=dry_run,
        verbose=verbose,
        prefer_cpu=prefer_cpu,
    )


def delegate_transformer_evaluate(
    *,
    config_path: Path,
    run_id: str,
    checkpoint: Path | None,
    output_dir: Path | None,
    beam_size: int,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """Déléguer l'évaluation SacreBLEU à ``1_Transformer/5_evaluate.py``."""
    eval_module = _load_transformer_module("5_evaluate.py", "s3t_multimodal_evaluate")
    code = eval_module.run_evaluate(
        config_path=resolve_config_path(config_path),
        run_id=run_id,
        checkpoint=checkpoint,
        output_dir=output_dir,
        beam_size=beam_size,
        dry_run=dry_run,
        verbose=verbose,
        prefer_cpu=prefer_cpu,
    )
    if code != 0 or dry_run:
        return code

    config = load_yaml_config(resolve_config_path(config_path))
    run_dir = resolve_run_dir(config, run_id=run_id, output_dir_override=output_dir)
    metrics_path = run_dir / "eval" / "metrics.json"
    if metrics_path.is_file():
        payload: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        payload["pipeline"] = PIPELINE_NAME
        write_json(metrics_path, payload)

    try:
        from scripts_communs.update_experiments_tracking import sync_run_from_metrics

        if sync_run_from_metrics(run_dir) and verbose:
            print(f"  Tracking:  runs/experiments_tracking.csv (run_id={run_id})")
    except OSError as exc:
        print(f"WARNING: tracking CSV not updated: {exc}", file=sys.stderr)

    return code


def delegate_transformer_infer(
    *,
    checkpoint: Path,
    input_audio: Path,
    config_path: Path | None,
    beam_size: int,
    output: Path,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """Déléguer l'inférence WAV à ``1_Transformer/6_infer.py``."""
    infer_module = _load_transformer_module("6_infer.py", "s3t_multimodal_infer")
    return infer_module.run_infer(
        checkpoint=checkpoint,
        input_audio=input_audio,
        config_path=resolve_config_path(config_path) if config_path else None,
        beam_size=beam_size,
        output=output,
        dry_run=dry_run,
        verbose=verbose,
        prefer_cpu=prefer_cpu,
    )
