#!/usr/bin/env python3
"""
Mise à jour de ``runs/experiments_tracking.csv`` à partir des ``eval/metrics.json``.

Chaque run Gemini ou speechLLM peut être résumé avec BLEU, durée et coût estimé
(API Gemini ou GPU local). Ce module est appelé en fin d'évaluation Gemini et
peut être lancé manuellement pour resynchroniser tout le dépôt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_communs.st_common import deep_get, load_yaml_config  # noqa: E402

TRACKING_PATH = PROJECT_ROOT / "runs" / "experiments_tracking.csv"

# Grille par défaut Gemini 2.5 Flash (Developer API, entrée audio) — juin 2026.
# Source : https://ai.google.dev/gemini-api/docs/pricing
DEFAULT_GEMINI_FLASH_INPUT_USD_PER_1M = 1.0
DEFAULT_GEMINI_FLASH_OUTPUT_USD_PER_1M = 2.5

TRACKING_COLUMNS = [
    "run_id",
    "lang_pair",
    "pipeline",
    "segment_mode",
    "seed",
    "freeze_updates",
    "vocab_size",
    "beam",
    "bleu_dev",
    "bleu_test",
    "chrf_dev",
    "chrf_test",
    "ter_dev",
    "ter_test",
    "train_hours",
    "gpu_hours",
    "estimated_gpu_cost_usd",
    "max_gpu_mem_gb",
    "gemini_duration_min",
    "gemini_cost_usd",
    "gemini_input_usd_per_1m",
    "gemini_output_usd_per_1m",
    "git_commit",
    "status",
    "notes",
]


def _fmt_float(value: float | int | None, *, digits: int = 4) -> str:
    """Formater un nombre flottant pour le CSV (vide si absent)."""
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _infer_segment_mode(config: dict[str, Any] | None) -> str:
    """Déduire utterance vs sentence_like depuis la config ou les chemins manifests."""
    if config is None:
        return ""
    explicit = deep_get(config, "data.segment_mode", "")
    if explicit:
        return str(explicit)
    for key in ("data.train_manifest", "data.valid_manifest", "data.test_manifest"):
        path = str(deep_get(config, key, ""))
        if "manifests_sentence" in path:
            return "sentence_like"
        if "manifests/" in path:
            return "utterance"
    return ""


def _load_metrics_row(metrics_path: Path) -> dict[str, str]:
    """
    Construire une ligne CSV à partir d'un ``eval/metrics.json`` ou ``metrics.json`` train.

    Paramètres :
        metrics_path : Chemin vers le fichier JSON du run.

    Retour :
        Dictionnaire colonne → valeur (chaînes) pour ``experiments_tracking.csv``.
    """
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    pipeline = str(payload.get("pipeline", ""))
    run_id = str(payload.get("run_id", metrics_path.parent.parent.name))
    config: dict[str, Any] | None = payload.get("config")
    if isinstance(config, str):
        config_path = Path(config)
        config = load_yaml_config(config_path) if config_path.is_file() else None
    elif isinstance(config, dict):
        config_path = None
    else:
        config_path = None
        config = None

    lang_pair = ""
    if isinstance(config, dict):
        lang_pair = str(deep_get(config, "experiment.lang_pair", ""))
    if not lang_pair and "fr-en" in run_id:
        lang_pair = "fr-en"

    row: dict[str, str] = {col: "" for col in TRACKING_COLUMNS}
    row["run_id"] = run_id
    row["lang_pair"] = lang_pair
    row["pipeline"] = pipeline
    row["segment_mode"] = _infer_segment_mode(config)
    row["status"] = "ok"

    dev = payload.get("dev") or {}
    test = payload.get("test") or {}
    if isinstance(dev, dict):
        row["bleu_dev"] = _fmt_float(dev.get("bleu"), digits=2)
        row["chrf_dev"] = _fmt_float(dev.get("chrf"), digits=2)
        row["ter_dev"] = _fmt_float(dev.get("ter"), digits=2)
    if isinstance(test, dict):
        row["bleu_test"] = _fmt_float(test.get("bleu"), digits=2)
        row["chrf_test"] = _fmt_float(test.get("chrf"), digits=2)
        row["ter_test"] = _fmt_float(test.get("ter"), digits=2)

    row["beam"] = _fmt_float(payload.get("beam_size"), digits=0)
    row["git_commit"] = str(payload.get("git_commit", ""))

    if pipeline == "gemini_st":
        runtime = payload.get("runtime") or {}
        cost_block = payload.get("gemini_cost_estimate_usd") or {}
        row["gemini_duration_min"] = _fmt_float(
            runtime.get("elapsed_minutes"), digits=2
        )
        row["gemini_cost_usd"] = _fmt_float(cost_block.get("total"), digits=4)
        row["gemini_input_usd_per_1m"] = _fmt_float(
            cost_block.get("input_per_1m_tokens_usd")
            or DEFAULT_GEMINI_FLASH_INPUT_USD_PER_1M,
            digits=2,
        )
        row["gemini_output_usd_per_1m"] = _fmt_float(
            cost_block.get("output_per_1m_tokens_usd")
            or DEFAULT_GEMINI_FLASH_OUTPUT_USD_PER_1M,
            digits=2,
        )
        model_id = str(payload.get("model_id", ""))
        limit = deep_get(payload, "decode.limit", 0)
        notes = f"model={model_id}"
        if limit:
            notes += f"; limit={limit}"
        sig = dev.get("signature") if isinstance(dev, dict) else ""
        if sig:
            notes += f"; sacrebleu_signature={sig}"
        row["notes"] = notes

    elif pipeline in ("speechllm", "pantagruel_multimodal"):
        row["gpu_hours"] = _fmt_float(payload.get("gpu_hours"), digits=3)
        row["estimated_gpu_cost_usd"] = _fmt_float(
            payload.get("estimated_cost_usd"), digits=4
        )
        if isinstance(config, dict):
            row["seed"] = str(deep_get(config, "experiment.seed", ""))
            row["freeze_updates"] = str(
                deep_get(config, "train.freeze_encoder_updates", "")
            )
            enc = str(deep_get(config, "model.encoder_name", ""))
            if enc:
                row["notes"] = enc.split("/")[-1]
        checkpoint = str(payload.get("checkpoint", ""))
        if checkpoint:
            row["notes"] = (
                f"{row['notes']}; ckpt={Path(checkpoint).name}"
                if row["notes"]
                else f"checkpoint={Path(checkpoint).name}"
            )

    return row


def upsert_tracking_row(
    row: dict[str, str],
    *,
    tracking_path: Path = TRACKING_PATH,
) -> None:
    """
    Insérer ou remplacer une ligne dans ``experiments_tracking.csv``.

    Paramètres :
        row : Valeurs par colonne (clés = ``TRACKING_COLUMNS``).
        tracking_path : Fichier CSV agrégé (créé si absent).
    """
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if tracking_path.is_file():
        with tracking_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for existing in reader:
                if existing.get("run_id") != row.get("run_id"):
                    rows.append(
                        {col: existing.get(col, "") or "" for col in TRACKING_COLUMNS}
                    )
    rows.append({col: row.get(col, "") for col in TRACKING_COLUMNS})
    with tracking_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=TRACKING_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def sync_run_from_metrics(
    run_dir: Path,
    *,
    tracking_path: Path = TRACKING_PATH,
) -> bool:
    """
    Mettre à jour le tracking depuis ``<run_dir>/eval/metrics.json`` ou ``metrics.json``.

    Retour :
        True si une ligne a été écrite, False si aucun metrics trouvé.
    """
    for candidate in (run_dir / "eval" / "metrics.json", run_dir / "metrics.json"):
        if candidate.is_file():
            row = _load_metrics_row(candidate)
            upsert_tracking_row(row, tracking_path=tracking_path)
            return True
    return False


def sync_all_runs(
    runs_root: Path = PROJECT_ROOT / "runs",
    *,
    tracking_path: Path = TRACKING_PATH,
) -> int:
    """
    Resynchroniser toutes les lignes depuis les ``eval/metrics.json`` sous ``runs/``.

    Retour :
        Nombre de runs mis à jour.
    """
    count = 0
    for metrics_path in sorted(runs_root.glob("**/eval/metrics.json")):
        row = _load_metrics_row(metrics_path)
        upsert_tracking_row(row, tracking_path=tracking_path)
        count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    """Parseur CLI pour la synchronisation manuelle du CSV de tracking."""
    parser = argparse.ArgumentParser(
        description="Mettre à jour runs/experiments_tracking.csv depuis metrics.json",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Dossier run (ex. runs/fr-en/run_001_...). Sinon --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Resynchroniser tous les metrics.json sous runs/",
    )
    parser.add_argument(
        "--tracking-csv",
        type=Path,
        default=TRACKING_PATH,
        help="Chemin du CSV agrégé",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI."""
    args = build_parser().parse_args(argv)
    tracking_path = args.tracking_csv
    if args.all:
        count = sync_all_runs(tracking_path=tracking_path)
        print(f"Synchronisé {count} run(s) → {tracking_path}")
        return 0
    if args.run_dir is None:
        print("ERROR: specify --run-dir or --all", file=sys.stderr)
        return 2
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if sync_run_from_metrics(run_dir, tracking_path=tracking_path):
        print(f"Tracking mis à jour pour {run_dir.name} → {tracking_path}")
        return 0
    print(f"ERROR: no metrics.json under {run_dir}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
