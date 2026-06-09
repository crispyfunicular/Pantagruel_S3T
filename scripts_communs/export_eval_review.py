#!/usr/bin/env python3
"""
Exporter un TSV de relecture humaine : id, français (src), référence TEDx (EN), hypothèse modèle.

Les fichiers ``*_predictions.txt`` (une ligne par segment) sont fragiles si l'hypothèse
contient des retours à la ligne. Préférer ``eval/*_review.tsv`` produit par ``evaluate``.

Usage :
    python scripts_communs/export_eval_review.py \\
        --run-dir runs/fr-en/run_001_gemini_flash_sentence_like_v2 \\
        --split dev

    python scripts_communs/export_eval_review.py \\
        --manifest datasets/manifests_sentence/fr-en/valid.tsv \\
        --output review_valid_refs_only.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_communs.config_utils import deep_get, load_yaml_config  # noqa: E402

REVIEW_COLUMNS = (
    "id",
    "audio",
    "src_lang",
    "src_text",
    "tgt_lang",
    "tgt_text",
    "hypothesis",
    "notes",
)


def read_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    """Lire toutes les colonnes d'un manifest TSV prepare."""
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(
                f"Manifest invalide (colonne id manquante): {manifest_path}"
            )
        return [dict(row) for row in reader]


def read_predictions_lines(pred_path: Path, expected: int) -> list[str]:
    """
    Charger les hypothèses depuis un fichier ligne-à-ligne.

    Lève ValueError si le décompte ne correspond pas (souvent à cause de \\n dans les hyp).
    """
    if not pred_path.is_file():
        raise FileNotFoundError(pred_path)
    text = pred_path.read_text(encoding="utf-8")
    if text.endswith("\n"):
        text = text[:-1]
    lines = text.split("\n") if text else []
    if len(lines) == expected:
        return lines
    if len(lines) == expected + 1 and not lines[-1].strip():
        return lines[:-1]
    raise ValueError(
        f"Décalage prédictions/manifest: {len(lines)} lignes dans {pred_path.name}, "
        f"{expected} segments dans le manifest. "
        "Relancez evaluate (génère *_review.tsv) ou utilisez un export existant."
    )


def load_hypotheses(
    eval_dir: Path,
    split: str,
    *,
    expected: int,
) -> list[str]:
    """Charger les hypothèses depuis review.tsv, jsonl ou predictions.txt."""
    review_tsv = eval_dir / f"{split}_review.tsv"
    if review_tsv.is_file():
        hyps: list[str] = []
        with review_tsv.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                hyps.append((row.get("hypothesis") or "").strip())
        if len(hyps) != expected:
            raise ValueError(
                f"{review_tsv.name}: {len(hyps)} lignes, manifest {expected}"
            )
        return hyps

    jsonl_path = eval_dir / f"{split}_predictions.jsonl"
    if jsonl_path.is_file():
        hyps = []
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    hyps.append(str(json.loads(line).get("hypothesis", "")).strip())
        if len(hyps) != expected:
            raise ValueError(
                f"{jsonl_path.name}: {len(hyps)} lignes, manifest {expected}"
            )
        return hyps

    return read_predictions_lines(eval_dir / f"{split}_predictions.txt", expected)


def write_review_tsv(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    """Écrire le TSV de relecture (champs multi-lignes échappés correctement)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=REVIEW_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in REVIEW_COLUMNS})


def build_review_rows(
    manifest_rows: list[dict[str, str]],
    hypotheses: list[str] | None,
) -> list[dict[str, str]]:
    """Assembler les lignes de relecture."""
    out: list[dict[str, str]] = []
    for index, mrow in enumerate(manifest_rows):
        hyp = ""
        if hypotheses is not None:
            if index >= len(hypotheses):
                break
            hyp = hypotheses[index]
        out.append(
            {
                "id": mrow.get("id", ""),
                "audio": mrow.get("audio", ""),
                "src_lang": mrow.get("src_lang", "fr"),
                "src_text": (mrow.get("src_text") or "").strip(),
                "tgt_lang": mrow.get("tgt_lang", "en"),
                "tgt_text": (mrow.get("tgt_text") or "").strip(),
                "hypothesis": hyp,
                "notes": "",
            }
        )
    return out


def write_eval_review_artifacts(
    eval_dir: Path,
    split: str,
    manifest_path: Path,
    predictions: list[str],
) -> Path:
    """
    Écrire ``<split>_review.tsv`` et ``<split>_predictions.jsonl`` alignés sur le manifest.

    Retour :
        Chemin du TSV de relecture.
    """
    manifest_rows = read_manifest_rows(manifest_path)
    if len(manifest_rows) != len(predictions):
        raise ValueError(
            f"Review export: {len(predictions)} hypothèses, "
            f"{len(manifest_rows)} lignes dans {manifest_path.name}"
        )
    rows = build_review_rows(manifest_rows, predictions)
    review_path = eval_dir / f"{split}_review.tsv"
    write_review_tsv(review_path, rows)
    jsonl_path = eval_dir / f"{split}_predictions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {"id": row["id"], "hypothesis": row["hypothesis"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return review_path


def resolve_manifest_from_run(run_dir: Path, split: str) -> Path:
    """Déduire le manifest depuis eval/metrics.json ou config.yaml du run."""
    manifest_key = "data.valid_manifest" if split == "dev" else "data.test_manifest"
    metrics_path = run_dir / "eval" / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = payload.get("config")
        if isinstance(config, dict):
            rel = deep_get(config, manifest_key, "")
            if rel:
                return PROJECT_ROOT / str(rel)
        config_str = payload.get("config")
        if isinstance(config_str, str) and config_str:
            cfg = load_yaml_config(Path(config_str))
            rel = deep_get(cfg, manifest_key, "")
            if rel:
                return PROJECT_ROOT / str(rel)

    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        cfg = load_yaml_config(config_path)
        rel = deep_get(cfg, manifest_key, "")
        if rel:
            return PROJECT_ROOT / str(rel)

    raise FileNotFoundError(
        f"Impossible de trouver le manifest pour {run_dir} (split={split})"
    )


def export_review(
    *,
    manifest_path: Path,
    eval_dir: Path | None,
    split: str,
    output_path: Path,
    refs_only: bool,
) -> int:
    """Construire le TSV de relecture."""
    manifest_rows = read_manifest_rows(manifest_path)
    hypotheses: list[str] | None = None
    if not refs_only:
        if eval_dir is None:
            print(
                "ERROR: --refs-only ou --run-dir requis pour les hypothèses",
                file=sys.stderr,
            )
            return 2
        try:
            hypotheses = load_hypotheses(eval_dir, split, expected=len(manifest_rows))
        except (ValueError, FileNotFoundError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print(
                "Astuce: relancez evaluate (Gemini ou speechLLM) après mise à jour du dépôt ; "
                "cela crée eval/dev_review.tsv aligné.",
                file=sys.stderr,
            )
            return 2

    rows = build_review_rows(manifest_rows, hypotheses)
    write_review_tsv(output_path, rows)
    print(f"Écrit {len(rows)} lignes → {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Parseur CLI."""
    parser = argparse.ArgumentParser(description="Export TSV relecture humaine ST")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Dossier run (ex. runs/fr-en/run_001_gemini_...)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest TSV (si pas --run-dir)",
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test"),
        default="dev",
        help="Split (dev=valid, test=test)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Fichier TSV de sortie",
    )
    parser.add_argument(
        "--refs-only",
        action="store_true",
        help="Exporter seulement FR + référence EN (sans hypothèse)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI."""
    args = build_parser().parse_args(argv)
    if args.run_dir is not None:
        run_dir = (
            args.run_dir if args.run_dir.is_absolute() else PROJECT_ROOT / args.run_dir
        )
        manifest_path = resolve_manifest_from_run(run_dir, args.split)
        eval_dir = run_dir / "eval"
        default_out = eval_dir / f"{args.split}_review.tsv"
    elif args.manifest is not None:
        manifest_path = (
            args.manifest
            if args.manifest.is_absolute()
            else PROJECT_ROOT / args.manifest
        )
        eval_dir = None
        default_out = manifest_path.with_name(f"{manifest_path.stem}_review.tsv")
    else:
        print("ERROR: --run-dir ou --manifest requis", file=sys.stderr)
        return 2

    output_path = args.output or default_out
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    return export_review(
        manifest_path=manifest_path,
        eval_dir=eval_dir,
        split=args.split,
        output_path=output_path,
        refs_only=bool(args.refs_only),
    )


if __name__ == "__main__":
    sys.exit(main())
