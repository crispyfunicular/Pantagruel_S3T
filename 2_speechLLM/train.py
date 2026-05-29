#!/usr/bin/env python3
"""
Entraînement speechLLM B1 — projecteur seul (Pantagruel + LLM gelés).

Entrées : config YAML, manifests m-TEDx.
Sorties : ``runs/.../checkpoints/{best,last}.pt``, ``train.log``, ``metrics.json``.
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
from speechLLM.speechllm_common import (
    PROJECT_ROOT,
    SpeechLLMModel,
    collate_speechllm_batch,
    deep_get,
    load_speechllm_from_config,
    load_yaml_config,
    read_manifest,
    resolve_run_dir,
    resolve_speechllm_config_path,
    save_projector_checkpoint,
    set_seed,
    write_json,
)
from torch.utils.data import DataLoader


@dataclass
class TrainLogEvent:
    timestamp_utc: str
    update: int
    train_loss: float
    bleu_dev: float | None
    best_bleu_dev: float
    lr: float


def get_device(prefer_cpu: bool) -> torch.device:
    """Choisir CUDA si disponible sauf si `--prefer-cpu` est activé."""
    if not prefer_cpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_bleu(
    *,
    model: SpeechLLMModel,
    valid_loader: DataLoader,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    max_eval_batches: int | None,
) -> float:
    """Décoder `valid` et retourner le BLEU corpus (SacreBLEU)."""
    predictions: list[str] = []
    references: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            if max_eval_batches is not None and batch_idx >= max_eval_batches:
                break
            hyps = model.generate_text_batch(
                batch["input_values"].to(device),
                batch["attention_mask"].to(device),
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            predictions.extend(hyps)
            references.extend(batch["target_texts"])
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
    Entraîner le projecteur speechLLM (B1) et écrire les artifacts de run.

    Le checkpoint `best.pt` est sélectionné sur le BLEU dev afin de rester comparable à la
    baseline ST du dépôt.
    """
    config_path = resolve_speechllm_config_path(config_path)
    config = load_yaml_config(config_path)
    run_dir = resolve_run_dir(config, run_id=run_id, output_dir_override=output_dir)
    checkpoints_dir = run_dir / "checkpoints"
    train_log_path = run_dir / "train.log"
    metrics_path = run_dir / "metrics.json"

    train_manifest = PROJECT_ROOT / str(deep_get(config, "data.train_manifest"))
    valid_manifest = PROJECT_ROOT / str(deep_get(config, "data.valid_manifest"))
    sample_rate = int(deep_get(config, "data.sample_rate", 16000))
    prompt = str(
        deep_get(
            config,
            "prompt.template",
            "Translate the French speech to English.",
        )
    )

    seed = int(deep_get(config, "experiment.seed", 42))
    deterministic = bool(deep_get(config, "experiment.deterministic", True))
    max_updates = int(deep_get(config, "train.max_updates", 500))
    learning_rate = float(deep_get(config, "train.learning_rate_peak", 1e-4))
    weight_decay = float(deep_get(config, "train.weight_decay", 0.0))
    batch_size = int(deep_get(config, "train.batch_size", 2))
    grad_accum = int(deep_get(config, "train.gradient_accumulation", 1))
    grad_clip = float(deep_get(config, "train.gradient_clip_norm", 1.0))
    eval_every = int(deep_get(config, "train.eval_every_updates", 100))
    warmup_updates = int(deep_get(config, "train.warmup_updates", 1000))
    max_eval_batches = deep_get(config, "train.max_eval_batches", 20)
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    max_target_tokens = int(deep_get(config, "train.max_target_tokens", 256))
    max_new_tokens = int(deep_get(config, "decode.max_new_tokens", 128))
    num_beams = int(deep_get(config, "decode.beam_size", 1))
    amp_dtype = str(deep_get(config, "train.amp_dtype", "fp16")).lower()

    if not train_manifest.is_file() or not valid_manifest.is_file():
        print("ERROR: missing train/valid manifest", file=sys.stderr)
        return 2

    if dry_run:
        print("[dry-run] speechLLM train:")
        print(f"  config:   {config_path}")
        print(f"  run_dir:  {run_dir}")
        print(f"  llm:      {deep_get(config, 'model.llm_name')}")
        return 0

    start_wall_s = time.time()
    start_utc = datetime.now(timezone.utc).isoformat()

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
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

    train_samples = read_manifest(train_manifest)
    valid_samples = read_manifest(valid_manifest)
    if not train_samples or not valid_samples:
        print("ERROR: empty train/valid manifest", file=sys.stderr)
        return 2

    def collate_fn(batch: list) -> dict:
        """Collate DataLoader : audio paddé + textes cibles bruts (speechLLM)."""
        return collate_speechllm_batch(batch, sample_rate=sample_rate)

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

    model = load_speechllm_from_config(config, device=device)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    amp_enabled = device.type == "cuda"
    amp_type = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    best_bleu = -1.0
    global_update = 0
    accumulated = 0
    stop_training = False

    def current_lr() -> float:
        """Scheduler simplifié : warmup linéaire puis plateau."""
        if global_update < warmup_updates:
            return (
                learning_rate * float(global_update + 1) / float(max(warmup_updates, 1))
            )
        return learning_rate

    while not stop_training:
        for batch in train_loader:
            if global_update >= max_updates:
                stop_training = True
                break

            model.train()
            for group in optimizer.param_groups:
                group["lr"] = current_lr()

            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_texts = batch["target_texts"]

            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=amp_type,
            ):
                loss = model.forward_train(
                    input_values,
                    attention_mask,
                    target_texts,
                    prompt=prompt,
                    max_target_tokens=max_target_tokens,
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accumulated += 1

            if accumulated >= grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(),
                    max_norm=grad_clip,
                )
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
                        device=device,
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        max_eval_batches=max_eval_batches,
                    )
                    if bleu_dev > best_bleu:
                        best_bleu = bleu_dev
                        save_projector_checkpoint(
                            path=checkpoints_dir / "best.pt",
                            model=model,
                            config=config,
                            run_id=run_id,
                            git_commit=git_commit,
                            update=global_update,
                            best_bleu_dev=best_bleu,
                        )

                event = TrainLogEvent(
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    update=global_update,
                    train_loss=float(loss.item() * grad_accum),
                    bleu_dev=bleu_dev,
                    best_bleu_dev=best_bleu if best_bleu >= 0 else 0.0,
                    lr=float(optimizer.param_groups[0]["lr"]),
                )
                with train_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

                if verbose and (
                    global_update == 1
                    or global_update % 10 == 0
                    or global_update == max_updates
                ):
                    msg = f"update={global_update} loss={event.train_loss:.4f}"
                    if bleu_dev is not None:
                        msg += (
                            f" bleu_dev={bleu_dev:.2f} best={event.best_bleu_dev:.2f}"
                        )
                    print(msg)

    save_projector_checkpoint(
        path=checkpoints_dir / "last.pt",
        model=model,
        config=config,
        run_id=run_id,
        git_commit=git_commit,
        update=global_update,
        best_bleu_dev=best_bleu if best_bleu >= 0 else 0.0,
    )
    if not (checkpoints_dir / "best.pt").is_file():
        shutil.copy2(checkpoints_dir / "last.pt", checkpoints_dir / "best.pt")

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
            "pipeline": "speechllm",
            "run_id": run_id,
            "config_path": str(config_path.resolve()),
            "run_dir": str(run_dir.resolve()),
            "device": str(device),
            "config": config,
            "git_commit": git_commit,
            "updates": global_update,
            "best_bleu_dev": float(best_bleu if best_bleu >= 0 else 0.0),
        },
    )
    print(f"speechLLM training complete: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur CLI de l'étape `train` speechLLM."""
    parser = argparse.ArgumentParser(description="speechLLM — entraînement B1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefer-cpu", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par `2_speechLLM/pipeline.py train`."""
    return run_train(
        config_path=args.config,
        run_id=args.run_id,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        prefer_cpu=args.prefer_cpu,
    )


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    args = build_parser().parse_args(argv)
    return run_train(
        config_path=args.config,
        run_id=args.run_id,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        prefer_cpu=args.prefer_cpu,
    )


if __name__ == "__main__":
    sys.exit(main())
