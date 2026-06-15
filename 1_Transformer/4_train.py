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
from typing import Any

import sacrebleu
import torch
from scripts_communs.st_common import (
    PROJECT_ROOT,
    S3TModel,
    apply_feature_freq_mask,
    apply_waveform_time_mask,
    build_s3t_model,
    collate_for_training,
    decode_ids_to_text,
    deep_get,
    greedy_decode_batch,
    load_sentencepiece,
    load_yaml_config,
    parse_spec_augment_config,
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


def build_checkpoint_payload(
    *,
    run_id: str,
    config: dict[str, Any],
    model: S3TModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    vocab_size: int,
    git_commit: str,
    global_update: int,
    best_bleu: float,
    evals_without_improvement: int,
    start_utc: str,
) -> dict[str, Any]:
    """Assembler le dict sérialisé pour ``best.pt`` / ``last.pt`` (reprise incluse)."""
    return {
        "run_id": run_id,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "vocab_size": vocab_size,
        "git_commit": git_commit,
        "update": global_update,
        "best_bleu_dev": best_bleu,
        "evals_without_improvement": evals_without_improvement,
        "start_timestamp_utc": start_utc,
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Écrire un checkpoint sur disque."""
    torch.save(payload, path)


def resolve_resume_checkpoint(
    checkpoints_dir: Path,
    resume_from: Path | None,
) -> Path | None:
    """
    Choisir le checkpoint de reprise : explicite, sinon ``last.pt``, sinon ``best.pt``.

    Retour :
        Chemin existant, ou ``None`` si aucun checkpoint utilisable.
    """
    if resume_from is not None:
        if not resume_from.is_file():
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_from}")
        return resume_from
    for candidate in (checkpoints_dir / "last.pt", checkpoints_dir / "best.pt"):
        if candidate.is_file():
            return candidate
    return None


def load_train_checkpoint(
    path: Path,
    *,
    model: S3TModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> dict[str, Any]:
    """
    Restaurer poids (et optimiseur/scaler si présents) depuis un checkpoint étape 4.

    Retour :
        Métadonnées utiles à la boucle (update, best BLEU, early-stop counter, start UTC).
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid checkpoint payload: {path}")

    model.load_state_dict(payload["model_state"], strict=False)
    optimizer_state = payload.get("optimizer_state")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
    scaler_state = payload.get("scaler_state")
    if isinstance(scaler_state, dict) and scaler.is_enabled():
        scaler.load_state_dict(scaler_state)

    global_update = int(payload.get("update", 0))
    best_bleu = float(payload.get("best_bleu_dev", -1.0))
    evals_without_improvement = int(payload.get("evals_without_improvement", 0))
    start_utc = str(
        payload.get("start_timestamp_utc", datetime.now(timezone.utc).isoformat())
    )
    return {
        "global_update": global_update,
        "best_bleu": best_bleu,
        "evals_without_improvement": evals_without_improvement,
        "start_utc": start_utc,
        "checkpoint_path": str(path.resolve()),
    }


def run_train(
    *,
    config_path: Path,
    run_id: str,
    output_dir: Path | None,
    dry_run: bool,
    verbose: bool,
    prefer_cpu: bool,
    resume: bool = False,
    resume_from: Path | None = None,
    overwrite: bool = False,
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
        resume : Reprendre depuis ``last.pt`` / ``best.pt`` (ou ``resume_from``).
        resume_from : Checkpoint explicite pour la reprise.
        overwrite : Autoriser un entraînement neuf alors qu'un checkpoint existe déjà.

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
    warmup_updates = int(deep_get(config, "train.warmup_updates", 0))
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
    # 0 = désactivé ; sinon nombre d'évals dev consécutives sans gain BLEU avant arrêt (PRD §9).
    early_stopping_patience = int(deep_get(config, "train.early_stopping_patience", 0))
    spec_augment = parse_spec_augment_config(config)

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
        print(f"  warmup_updates: {warmup_updates}")
        print(f"  early_stopping_patience: {early_stopping_patience}")
        print(f"  resume:      {resume}")
        if resume_from is not None:
            print(f"  resume_from: {resume_from}")
        print(
            f"  spec_augment: enabled={spec_augment.enabled} "
            f"mask_time_prob={spec_augment.mask_time_prob} "
            f"mask_time_length={spec_augment.mask_time_length} "
            f"mask_freq_prob={spec_augment.mask_freq_prob} "
            f"mask_freq_length={spec_augment.mask_freq_length}"
        )
        return 0

    existing_checkpoint = resolve_resume_checkpoint(checkpoints_dir, None)
    if existing_checkpoint is not None and not resume and not overwrite:
        print(
            "ERROR: checkpoint existant sans --resume ni --overwrite : "
            f"{existing_checkpoint}",
            file=sys.stderr,
        )
        print(
            "  Utilisez --resume pour continuer, ou --overwrite pour repartir de zéro.",
            file=sys.stderr,
        )
        return 2

    start_wall_s = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
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
    evals_without_improvement = 0
    early_stopped = False

    if resume:
        checkpoint_path = resolve_resume_checkpoint(checkpoints_dir, resume_from)
        if checkpoint_path is None:
            print(
                "ERROR: --resume demandé mais aucun checkpoint trouvé", file=sys.stderr
            )
            return 2
        restored = load_train_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        global_update = restored["global_update"]
        best_bleu = restored["best_bleu"]
        evals_without_improvement = restored["evals_without_improvement"]
        start_utc = restored["start_utc"]
        if verbose:
            print(
                f"Resume from {checkpoint_path} "
                f"(update={global_update}, best_bleu_dev={best_bleu:.2f}, "
                f"evals_without_improvement={evals_without_improvement})"
            )
    elif overwrite and existing_checkpoint is not None:
        for stale in (checkpoints_dir / "best.pt", checkpoints_dir / "last.pt"):
            if stale.is_file():
                stale.unlink()
        if verbose:
            print(f"Overwrite: removed existing checkpoints in {checkpoints_dir}")

    amp_enabled = device.type == "cuda"
    amp_type = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    def current_lr(update: int) -> float:
        """
        Scheduler RF-10 : warmup linéaire puis décroissance inverse racine carrée.

        ``update`` est 1-indexé (premier pas optimiseur = 1).
        """
        if warmup_updates <= 0:
            return learning_rate
        if update <= warmup_updates:
            return learning_rate * float(update) / float(warmup_updates)
        return learning_rate * (float(warmup_updates) ** 0.5) / (float(update) ** 0.5)

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
            if spec_augment.enabled:
                input_values = apply_waveform_time_mask(
                    input_values,
                    attention_mask,
                    spec_augment=spec_augment,
                    sample_rate=sample_rate,
                )

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_type,
            ):
                memory = model.encode(input_values, attention_mask)
                if spec_augment.enabled:
                    memory = apply_feature_freq_mask(
                        memory,
                        spec_augment=spec_augment,
                    )
                logits = model.decode(memory, tokens_in)
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
                next_update = global_update + 1
                lr_now = current_lr(next_update)
                for group in optimizer.param_groups:
                    group["lr"] = lr_now
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
                        evals_without_improvement = 0
                        save_checkpoint(
                            checkpoints_dir / "best.pt",
                            build_checkpoint_payload(
                                run_id=run_id,
                                config=config,
                                model=model,
                                optimizer=optimizer,
                                scaler=scaler,
                                pad_id=pad_id,
                                bos_id=bos_id,
                                eos_id=eos_id,
                                vocab_size=vocab_size,
                                git_commit=git_commit,
                                global_update=global_update,
                                best_bleu=best_bleu,
                                evals_without_improvement=evals_without_improvement,
                                start_utc=start_utc,
                            ),
                        )
                    else:
                        evals_without_improvement += 1
                        if (
                            early_stopping_patience > 0
                            and evals_without_improvement >= early_stopping_patience
                        ):
                            early_stopped = True
                            stop_training = True
                            if verbose:
                                print(
                                    f"Early stopping at update={global_update} "
                                    f"(patience={early_stopping_patience}, "
                                    f"best_bleu_dev={best_bleu:.2f})"
                                )

                    # Sauvegarde périodique pour reprise après interruption (PRD : runs longs).
                    save_checkpoint(
                        checkpoints_dir / "last.pt",
                        build_checkpoint_payload(
                            run_id=run_id,
                            config=config,
                            model=model,
                            optimizer=optimizer,
                            scaler=scaler,
                            pad_id=pad_id,
                            bos_id=bos_id,
                            eos_id=eos_id,
                            vocab_size=vocab_size,
                            git_commit=git_commit,
                            global_update=global_update,
                            best_bleu=best_bleu,
                            evals_without_improvement=evals_without_improvement,
                            start_utc=start_utc,
                        ),
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

    save_checkpoint(
        checkpoints_dir / "last.pt",
        build_checkpoint_payload(
            run_id=run_id,
            config=config,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            vocab_size=vocab_size,
            git_commit=git_commit,
            global_update=global_update,
            best_bleu=best_bleu if best_bleu >= 0 else 0.0,
            evals_without_improvement=evals_without_improvement,
            start_utc=start_utc,
        ),
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
            "early_stopped": early_stopped,
            "early_stopping_patience": early_stopping_patience,
            "resumed": resume,
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre depuis checkpoints/last.pt ou best.pt",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint explicite pour --resume",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Repartir de zéro en supprimant les checkpoints existants",
    )
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
        resume=getattr(args, "resume", False),
        resume_from=getattr(args, "resume_from", None),
        overwrite=getattr(args, "overwrite", False),
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal. Code 0 si succès, 2 si erreur usage/données."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
