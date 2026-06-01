#!/usr/bin/env python3
"""
Étape 4 — Fine-tuning de la traduction vocale (encodeur Pantagruel + décodeur Transformer).

Entraîne la prédiction du token suivant sur m-TEDx avec teacher forcing, gel d'encodeur
optionnel, précision mixte et SacreBLEU périodique sur le split dev pour sélectionner ``best.pt``.

Entrées :
    - Config YAML (manifests, chemin SPM, hyperparamètres modèle et train).
    - Manifests TSV et WAV 16 kHz préparés à l'étape 2.
    - Checkpoint Hugging Face Pantagruel (téléchargé à la première utilisation).

Sorties (sous ``runs/<lang_pair>/<run_id>/``) :
    - ``checkpoints/best.pt``, ``checkpoints/last.pt``
    - ``train.log`` (lignes JSON par mise à jour)
    - ``metrics.json``, copie de ``config.yaml``

Codes de sortie : 0 succès, 2 entrées manquantes / erreur CLI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu
import torch
from scripts_communs.st_common import (
    PROJECT_ROOT,
    S3TModel,
    build_s3t_model,
    collate_for_training,
    decode_ids_to_text,
    deep_get,
    greedy_decode_batch,
    load_sentencepiece,
    load_yaml_config,
    read_manifest,
    resolve_run_dir,
    set_seed,
    write_json,
)
from torch.nn import functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainLogEvent:
    """Une ligne JSON dans ``train.log`` pour une étape optimiseur."""

    timestamp_utc: str
    update: int
    train_loss: float
    bleu_dev: float | None
    best_bleu_dev: float
    lr: float
    encoder_frozen: bool


def get_device(prefer_cpu: bool) -> torch.device:
    """Choisir CUDA si disponible sauf si ``--prefer-cpu`` est activé."""
    if not prefer_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_bleu(
    *,
    model: S3TModel,
    valid_loader: DataLoader,
    sp_model,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    max_eval_batches: int | None,
) -> float:
    """
    Décoder en glouton le jeu de validation et retourner le SacreBLEU corpus.

    Utilisé pendant l'entraînement pour choisir ``best.pt`` (PRD : BLEU dev plutôt que la perte).

    Paramètres :
        model : ``S3TModel`` actuel.
        valid_loader : DataLoader de validation.
        sp_model : SentencePiece pour la détokenisation.
        device : Périphérique Torch.
        bos_id, eos_id, pad_id: SPM special tokens.
        max_new_tokens : Plafond de longueur générée.
        max_eval_batches : Plafond optionnel pour éval mid-training plus rapide.

    Retour :
        Score SacreBLEU (0 si aucune référence).
    """
    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            if max_eval_batches is not None and batch_idx >= max_eval_batches:
                break
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = greedy_decode_batch(
                model=model,
                input_values=input_values,
                attention_mask=attention_mask,
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                max_new_tokens=max_new_tokens,
            ).cpu()
            targets = batch["tokens_out"]
            for row_idx in range(generated.size(0)):
                hyp = decode_ids_to_text(
                    generated[row_idx].tolist(),
                    sp_model=sp_model,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    pad_id=pad_id,
                )
                ref = decode_ids_to_text(
                    targets[row_idx].tolist(),
                    sp_model=sp_model,
                    bos_id=bos_id,
                    eos_id=eos_id,
                    pad_id=pad_id,
                )
                predictions.append(hyp)
                references.append(ref)
    if not references:
        return 0.0
    return float(sacrebleu.corpus_bleu(predictions, [references]).score)


def run_train(
    *,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
) -> int:
    """
    Exécuter la boucle d'entraînement ST complète depuis une config YAML.

    Paramètres :
        config_path : Config d'expérience (ex. ``1_Transformer/configs/fr-en/base.yaml``).
        run_id : Nom de sous-répertoire sous ``runs/<lang_pair>/``.
        output_dir : Surcharge du répertoire de run (optionnel).
        dry_run : Afficher le plan sans entraîner.
        verbose : Journaliser perte et BLEU périodiques.
        prefer_cpu : Forcer le CPU même si CUDA est disponible.

    Retour :
        0 on success, 2 if manifests/SPM are missing.
    """
    config = load_yaml_config(config_path)
    run_dir = resolve_run_dir(config, run_id=run_id, output_dir_override=output_dir)
    checkpoints_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    train_log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.json"

    train_manifest = PROJECT_ROOT / str(deep_get(config, "data.train_manifest"))
    valid_manifest = PROJECT_ROOT / str(deep_get(config, "data.valid_manifest"))
    spm_model_path = PROJECT_ROOT / str(deep_get(config, "data.spm_model"))
    sample_rate = int(deep_get(config, "data.sample_rate", 16000))

    encoder_name = str(
        deep_get(config, "model.encoder_name", "PantagrueLLM/Pantagruel-Base")
    )

    seed = int(deep_get(config, "experiment.seed", 42))
    deterministic = bool(deep_get(config, "experiment.deterministic", True))
    max_updates = int(deep_get(config, "train.max_updates", 500))
    freeze_encoder_updates = int(deep_get(config, "train.freeze_encoder_updates", 0))
    learning_rate = float(deep_get(config, "train.learning_rate_peak", 2e-4))
    weight_decay = float(deep_get(config, "train.weight_decay", 0.01))
    label_smoothing = float(deep_get(config, "train.label_smoothing", 0.1))
    batch_size = int(deep_get(config, "train.batch_size", 2))
    grad_accum = int(deep_get(config, "train.gradient_accumulation", 1))
    grad_clip = float(deep_get(config, "train.gradient_clip_norm", 1.0))
    eval_every = int(deep_get(config, "train.eval_every_updates", 100))
    max_eval_batches = deep_get(config, "train.max_eval_batches", 20)
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    max_target_tokens = int(deep_get(config, "train.max_target_tokens", 256))
    decode_max_new = int(deep_get(config, "decode.max_len_b", 128))
    amp_dtype = str(deep_get(config, "train.amp_dtype", "fp16")).lower()

    if not train_manifest.is_file() or not valid_manifest.is_file():
        print("ERROR: missing train/valid manifest in config", file=sys.stderr)
        return 2
    if not spm_model_path.is_file():
        print(f"ERROR: missing SentencePiece model: {spm_model_path}", file=sys.stderr)
        return 2

    if dry_run:
        print("[dry-run] train stage plan:")
        print(f"  config:      {config_path}")
        print(f"  run_id:      {run_id}")
        print(f"  run_dir:     {run_dir}")
        print(f"  manifests:   {train_manifest}, {valid_manifest}")
        print(f"  spm_model:   {spm_model_path}")
        print(f"  encoder:     {encoder_name}")
        print(f"  max_updates: {max_updates}")
        return 0

    start_wall_s = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")

    git_commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
        git_commit = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    set_seed(seed, deterministic)
    device = get_device(prefer_cpu=prefer_cpu)
    if verbose:
        print(f"Device: {device}")

    sp_model = load_sentencepiece(spm_model_path)
    pad_id = int(sp_model.pad_id())
    bos_id = int(sp_model.bos_id())
    eos_id = int(sp_model.eos_id())
    vocab_size = int(sp_model.get_piece_size())

    train_samples = read_manifest(train_manifest)
    valid_samples = read_manifest(valid_manifest)
    if not train_samples or not valid_samples:
        print("ERROR: empty train/valid manifest", file=sys.stderr)
        return 2

    def collate_fn(batch):
        """Collate DataLoader : audio paddé + tokens cibles (teacher forcing)."""
        return collate_for_training(
            batch,
            sp_model=sp_model,
            sample_rate=sample_rate,
            max_target_tokens=max_target_tokens,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )

    train_loader = DataLoader(
        train_samples,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_samples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = build_s3t_model(
        config,
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_positions=max_target_tokens + 2,
    ).to(device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_bleu = -1.0
    global_update = 0
    accumulated = 0
    train_events: list[TrainLogEvent] = []
    stop_training = False

    amp_enabled = device.type == "cuda"
    amp_type = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    while not stop_training:
        for batch in train_loader:
            if global_update >= max_updates:
                stop_training = True
                break

            model.train()
            # RF-11 : geler Pantagruel tôt pour qu'un décodeur aléatoire n'endommage pas les poids SSL.
            should_freeze = global_update < freeze_encoder_updates
            model.freeze_encoder(should_freeze)

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tokens_in = batch["tokens_in"].to(device)
            tokens_out = batch["tokens_out"].to(device)

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_type,
            ):
                logits = model(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    tokens=tokens_in,
                )
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens_out.reshape(-1),
                    ignore_index=pad_id,
                    label_smoothing=label_smoothing,
                )
                # Mettre à l'échelle la perte lors de l'accumulation de gradients sur micro-lots.
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accumulated += 1

            if accumulated >= grad_accum:
                # Une étape optimiseur par ``grad_accum`` passes avant.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                global_update += 1

                bleu_dev: float | None = None
                if global_update % eval_every == 0 or global_update == max_updates:
                    bleu_dev = evaluate_bleu(
                        model=model,
                        valid_loader=valid_loader,
                        sp_model=sp_model,
                        device=device,
                        bos_id=bos_id,
                        eos_id=eos_id,
                        pad_id=pad_id,
                        max_new_tokens=decode_max_new,
                        max_eval_batches=max_eval_batches,
                    )
                    if bleu_dev > best_bleu:
                        best_bleu = bleu_dev
                        torch.save(
                            {
                                "run_id": run_id,
                                "config": config,
                                "model_state": model.state_dict(),
                                "pad_id": pad_id,
                                "bos_id": bos_id,
                                "eos_id": eos_id,
                                "vocab_size": vocab_size,
                                "git_commit": git_commit,
                                "update": global_update,
                                "best_bleu_dev": best_bleu,
                            },
                            checkpoints_dir / "best.pt",
                        )

                event = TrainLogEvent(
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    update=global_update,
                    train_loss=float(loss.item() * grad_accum),
                    bleu_dev=bleu_dev,
                    best_bleu_dev=best_bleu if best_bleu >= 0 else 0.0,
                    lr=float(optimizer.param_groups[0]["lr"]),
                    encoder_frozen=should_freeze,
                )
                train_events.append(event)
                with train_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

                if verbose and (
                    global_update == 1
                    or global_update % 10 == 0
                    or global_update == max_updates
                ):
                    msg = (
                        f"update={global_update} loss={event.train_loss:.4f} "
                        f"freeze={event.encoder_frozen}"
                    )
                    if event.bleu_dev is not None:
                        msg += f" bleu_dev={event.bleu_dev:.2f} best={event.best_bleu_dev:.2f}"
                    print(msg)

    torch.save(
        {
            "run_id": run_id,
            "config": config,
            "model_state": model.state_dict(),
            "pad_id": pad_id,
            "bos_id": bos_id,
            "eos_id": eos_id,
            "vocab_size": vocab_size,
            "git_commit": git_commit,
            "update": global_update,
            "best_bleu_dev": best_bleu if best_bleu >= 0 else 0.0,
        },
        checkpoints_dir / "last.pt",
    )

    if not (checkpoints_dir / "best.pt").is_file():
        shutil.copy2(checkpoints_dir / "last.pt", checkpoints_dir / "best.pt")
        best_bleu = best_bleu if best_bleu >= 0 else 0.0

    write_json(
        metrics_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "start_timestamp_utc": start_utc,
            "end_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": float(time.time() - start_wall_s),
            "gpu_hours": float(time.time() - start_wall_s) / 3600.0
            if device.type == "cuda"
            else 0.0,
            "estimated_cost_usd": (
                float(time.time() - start_wall_s)
                / 3600.0
                * float(deep_get(config, "cost.usd_per_gpu_hour", 0.0))
                if device.type == "cuda"
                else 0.0
            ),
            "run_id": run_id,
            "config_path": str(config_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "device": str(device),
            "config": config,
            "git_commit": git_commit,
            "updates": global_update,
            "best_bleu_dev": float(best_bleu if best_bleu >= 0 else 0.0),
            "train_events": len(train_events),
            "checkpoints": {
                "best": str((checkpoints_dir / "best.pt").resolve()),
                "last": str((checkpoints_dir / "last.pt").resolve()),
            },
        },
    )
    print(f"Training complete. Run directory: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """CLI pour l'étape 4."""
    parser = argparse.ArgumentParser(
        description="S3T Étape 4 — Entraînement ST",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefer-cpu", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par ``pipeline.py train``."""
    config_path = getattr(args, "config", None)
    if config_path is None:
        print("ERROR: --config is required for train stage", file=sys.stderr)
        return 2
    run_id = getattr(args, "run_id", None)
    if run_id is None:
        print("ERROR: --run-id is required for train stage", file=sys.stderr)
        return 2
    return run_train(
        config_path=config_path,
        run_id=run_id,
        output_dir=getattr(args, "output_dir", None),
        dry_run=getattr(args, "dry_run", False),
        verbose=getattr(args, "verbose", False),
        prefer_cpu=getattr(args, "prefer_cpu", False),
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal. Code 0 si succès, 2 si erreur usage/données."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
