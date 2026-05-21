#!/usr/bin/env python3
"""
Étape 5 — Décoder les splits valid/test et scorer avec SacreBLEU (et CHRF, TER).

Charge ``best.pt`` (ou un checkpoint donné), exécute un décodage glouton sur les manifests
dev/test, écrit des fichiers de prédictions alignés ligne à ligne et stocke les artefacts
métriques canoniques pour comparaison avec le tableau 8 Pantagruel (RF-14–19).

Entrées :
    - Config YAML + checkpoint de l'étape 4.
    - Manifests TSV valid/test et modèle SPM.

Sorties (``runs/<lang_pair>/<run_id>/eval/``) :
    - ``dev_predictions.txt``, ``test_predictions.txt``
    - ``sacrebleu_dev.txt``, ``sacrebleu_test.txt`` (inclut la signature SacreBLEU)
    - ``metrics.json``

Note : ``--beam-size`` est journalisé mais le décodage utilise actuellement la recherche
gloutonne dans ``greedy_decode_batch`` (beam search prévu pour alignement article).

Codes de sortie : 0 succès, 2 entrées manquantes.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from scripts.st_common import (
    PROJECT_ROOT,
    S3TModel,
    collate_for_training,
    decode_ids_to_text,
    deep_get,
    greedy_decode_batch,
    load_sentencepiece,
    load_yaml_config,
    read_manifest,
    resolve_run_dir,
    write_json,
)
from torch.utils.data import DataLoader


def load_checkpoint(path: Path) -> dict[str, Any]:
    """
    Charger un checkpoint d'entraînement écrit par l'étape 4.

    Paramètres :
        path : ``best.pt`` ou ``last.pt``.

    Retour :
        Dict avec ``model_state``, ids de tokens, config embarquée optionnelle.

    Lève :
        FileNotFoundError, ValueError : Fichier invalide ou manquant.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")
    return payload


def decode_manifest(
    *,
    model: S3TModel,
    loader: DataLoader,
    sp_model,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
) -> tuple[list[str], list[str]]:
    """
    Décoder en glouton tous les échantillons d'un DataLoader et détokeniser en chaînes.

    Retour :
        (predictions, references) comme listes parallèles d'une ligne par utterance.
    """
    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            generated = greedy_decode_batch(
                model=model,
                input_values=batch["input_values"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                max_new_tokens=max_new_tokens,
            ).cpu()
            targets = batch["tokens_out"]
            for idx in range(generated.size(0)):
                predictions.append(
                    decode_ids_to_text(
                        generated[idx].tolist(),
                        sp_model=sp_model,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                    )
                )
                references.append(
                    decode_ids_to_text(
                        targets[idx].tolist(),
                        sp_model=sp_model,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                    )
                )
    return predictions, references


def score_with_sacrebleu(preds: list[str], refs: list[str]) -> dict[str, Any]:
    """
    Calculer BLEU, CHRF, TER corpus et la signature de protocole SacreBLEU.

    Paramètres :
        preds : Hypothèses système (une chaîne par ligne).
        refs : Traductions de référence (même longueur que preds).

    Retour :
        Dict avec scores numériques, chaînes métriques lisibles et ``signature``.
    """
    bleu_metric = sacrebleu.metrics.BLEU()
    chrf_metric = sacrebleu.metrics.CHRF()
    ter_metric = sacrebleu.metrics.TER()
    bleu = bleu_metric.corpus_score(preds, [refs])
    chrf = chrf_metric.corpus_score(preds, [refs])
    ter = ter_metric.corpus_score(preds, [refs])
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "ter": float(ter.score),
        "signature": str(bleu_metric.get_signature()),
        "bleu_text": str(bleu),
        "chrf_text": str(chrf),
        "ter_text": str(ter),
    }


def run_evaluate(
    *,
    config_path: Path,
    run_id: str,
    checkpoint: Path | None,
    beam_size: int,
    output_dir: Path | None,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """
    Exécuter l'évaluation sur les splits valid et test pour un run d'entraînement.

    Paramètres :
        config_path: Experiment YAML.
        run_id: Run identifier matching stage 4.
        checkpoint : Chemin optionnel ; défaut ``checkpoints/best.pt``.
        beam_size : Enregistré dans les métriques (décodage glouton aujourd'hui).
        output_dir: Override run root.
        dry_run, verbose, prefer_cpu : Drapeaux CLI.

    Retour :
        0 on success, 2 if manifests/SPM/checkpoint are missing.
    """
    config = load_yaml_config(config_path)
    run_dir = resolve_run_dir(config, run_id=run_id, output_dir_override=output_dir)
    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    checkpoint_path = checkpoint or (checkpoints_dir / "best.pt")

    valid_manifest = PROJECT_ROOT / str(deep_get(config, "data.valid_manifest"))
    test_manifest = PROJECT_ROOT / str(deep_get(config, "data.test_manifest"))
    spm_model_path = PROJECT_ROOT / str(deep_get(config, "data.spm_model"))
    sample_rate = int(deep_get(config, "data.sample_rate", 16000))
    max_target_tokens = int(deep_get(config, "train.max_target_tokens", 256))
    max_new_tokens = int(deep_get(config, "decode.max_len_b", 128))

    if dry_run:
        print("[dry-run] evaluate stage plan:")
        print(f"  run_dir: {run_dir}")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  valid_manifest: {valid_manifest}")
        print(f"  test_manifest: {test_manifest}")
        print(f"  spm_model: {spm_model_path}")
        return 0

    if not valid_manifest.is_file() or not test_manifest.is_file():
        print("ERROR: missing valid/test manifest", file=sys.stderr)
        return 2
    if not spm_model_path.is_file():
        print(f"ERROR: missing SentencePiece model: {spm_model_path}", file=sys.stderr)
        return 2

    payload = load_checkpoint(checkpoint_path)
    ckpt_config = payload.get("config", config)
    encoder_name = str(
        deep_get(ckpt_config, "model.encoder_name", "PantagrueLLM/Pantagruel-Base")
    )
    hidden_dim = int(deep_get(ckpt_config, "model.hidden_dim", 768))
    decoder_layers = int(deep_get(ckpt_config, "model.decoder_layers", 6))
    decoder_heads = int(deep_get(ckpt_config, "model.decoder_heads", 8))
    dropout = float(deep_get(ckpt_config, "model.dropout", 0.1))

    sp_model = load_sentencepiece(spm_model_path)
    pad_id = int(payload.get("pad_id", sp_model.pad_id()))
    bos_id = int(payload.get("bos_id", sp_model.bos_id()))
    eos_id = int(payload.get("eos_id", sp_model.eos_id()))
    vocab_size = int(payload.get("vocab_size", sp_model.get_piece_size()))

    def collate_fn(batch):
        return collate_for_training(
            batch,
            sp_model=sp_model,
            sample_rate=sample_rate,
            max_target_tokens=max_target_tokens,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )

    valid_loader = DataLoader(
        read_manifest(valid_manifest),
        batch_size=int(deep_get(config, "train.batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        read_manifest(test_manifest),
        batch_size=int(deep_get(config, "train.batch_size", 2)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device(
        "cpu" if prefer_cpu or not torch.cuda.is_available() else "cuda"
    )
    model = S3TModel(
        encoder_name=encoder_name,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        decoder_layers=decoder_layers,
        decoder_heads=decoder_heads,
        dropout=dropout,
        pad_id=pad_id,
        max_positions=max_target_tokens + 2,
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=False)

    eval_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Evaluating checkpoint: {checkpoint_path}")

    dev_preds, dev_refs = decode_manifest(
        model=model,
        loader=valid_loader,
        sp_model=sp_model,
        device=device,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
    )
    test_preds, test_refs = decode_manifest(
        model=model,
        loader=test_loader,
        sp_model=sp_model,
        device=device,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
    )

    dev_scores = score_with_sacrebleu(dev_preds, dev_refs)
    test_scores = score_with_sacrebleu(test_preds, test_refs)

    (eval_dir / "dev_predictions.txt").write_text(
        "\n".join(dev_preds) + ("\n" if dev_preds else ""), encoding="utf-8"
    )
    (eval_dir / "test_predictions.txt").write_text(
        "\n".join(test_preds) + ("\n" if test_preds else ""), encoding="utf-8"
    )
    (eval_dir / "sacrebleu_dev.txt").write_text(
        "\n".join(
            [
                f"BLEU = {dev_scores['bleu']:.2f}",
                dev_scores["bleu_text"],
                f"CHRF = {dev_scores['chrf']:.2f}",
                dev_scores["chrf_text"],
                f"TER = {dev_scores['ter']:.2f}",
                dev_scores["ter_text"],
                f"Signature: {dev_scores['signature']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (eval_dir / "sacrebleu_test.txt").write_text(
        "\n".join(
            [
                f"BLEU = {test_scores['bleu']:.2f}",
                test_scores["bleu_text"],
                f"CHRF = {test_scores['chrf']:.2f}",
                test_scores["chrf_text"],
                f"TER = {test_scores['ter']:.2f}",
                test_scores["ter_text"],
                f"Signature: {test_scores['signature']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    write_json(
        eval_dir / "metrics.json",
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "checkpoint": str(checkpoint_path.resolve()),
            "beam_size": beam_size,
            "dev": dev_scores,
            "test": test_scores,
        },
    )

    print("Evaluation complete.")
    print(f"  BLEU dev:  {dev_scores['bleu']:.2f}")
    print(f"  BLEU test: {test_scores['bleu']:.2f}")
    print(f"  Eval dir:  {eval_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI pour l'étape 5."""
    parser = argparse.ArgumentParser(
        description="S3T Étape 5 — Évaluer checkpoint avec SacreBLEU",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefer-cpu", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par ``pipeline.py evaluate``."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        print("ERROR: --config is required for evaluate stage", file=sys.stderr)
        return 2
    run_id = getattr(args, "run_id", None)
    if run_id is None:
        print("ERROR: --run-id is required for evaluate stage", file=sys.stderr)
        return 2
    return run_evaluate(
        config_path=config_path,
        run_id=run_id,
        checkpoint=getattr(args, "checkpoint", None),
        beam_size=getattr(args, "beam_size", 5),
        output_dir=getattr(args, "output_dir", None),
        dry_run=getattr(args, "dry_run", False),
        verbose=getattr(args, "verbose", False),
        prefer_cpu=getattr(args, "prefer_cpu", False),
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
