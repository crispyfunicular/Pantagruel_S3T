#!/usr/bin/env python3
"""
Bibliothèque partagée pour les étapes ST S3T 4–6 (train, evaluate, infer).

Ce module centralise le modèle ST bout en bout (encodeur Pantagruel HF + décodeur
Transformer), le chargement des données depuis des manifests TSV, le collationnement
par lot pour le teacher forcing et le décodage glouton autorégressif.

Entrées (typiques) :
    - Config YAML de run (chemins manifests, modèle SPM, hyperparamètres).
    - Manifests TSV avec colonnes : id, audio, tgt_text (voir ``read_manifest``).
    - Fichiers WAV mono 16 kHz et un ``.model`` SentencePiece entraîné.

Sorties (utilisées par les appelants) :
    - Logits ``S3TModel`` pendant l'entraînement.
    - Ids de tokens gloutons depuis ``greedy_decode_batch``.
    - Répertoires de run résolus sous ``runs/<lang_pair>/<run_id>/``.

Dépendances : PyTorch, transformers (checkpoint Pantagruel), sentencepiece, soundfile, PyYAML.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn as nn
import yaml
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ManifestSample:
    """Une ligne ST entraînement/éval : chemin audio français et texte de référence anglais."""

    sample_id: str
    audio_path: Path
    target_text: str


def set_seed(seed: int, deterministic: bool) -> None:
    """
    Initialiser Python, NumPy et PyTorch (CUDA inclus) pour des runs reproductibles.

    Paramètres :
        seed : Graine aléatoire de base depuis la config d'expérience.
        deterministic : Si True, désactiver l'autotuning cuDNN (plus lent mais plus stable).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def deep_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Lire une valeur de config imbriquée en notation pointée (ex. ``train.batch_size``).

    Paramètres :
        config : Dict imbriqué chargé depuis YAML.
        key : Chemin séparé par des points.
        default : Valeur renvoyée si un segment du chemin est absent.

    Retour :
        The value at ``key`` or ``default``.
    """
    cursor: Any = config
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def load_yaml_config(path: Path) -> dict[str, Any]:
    """
    Charger une config d'expérience YAML dans un dict simple.

    Paramètres :
        path : Chemin vers ``base.yaml`` ou équivalent.

    Retour :
        Parsed mapping.

    Lève :
        ValueError: If the root YAML node is not a mapping.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML object: {path}")
    return payload


def ensure_project_relative(path_like: str | Path) -> Path:
    """
    Résoudre les chemins de manifest relatifs à la racine projet S3T si non absolus.

    Paramètres :
        path_like : Chaîne de chemin depuis un manifest ou fichier de config.

    Retour :
        Absolute path if ``path_like`` was absolute, else ``PROJECT_ROOT / path_like``.
    """
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def read_manifest(path: Path) -> list[ManifestSample]:
    """
    Analyser un manifest TSV produit par l'étape ``2_prepare``.

    Paramètres :
        path : Fichier manifest (train/valid/test.tsv).

    Retour :
        List of samples with resolved audio paths.

    Lève :
        ValueError: If required columns are missing.
    """
    import csv

    rows: list[ManifestSample] = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"id", "audio", "tgt_text"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Missing required columns in manifest: {path}")
        for row in reader:
            rows.append(
                ManifestSample(
                    sample_id=row["id"],
                    audio_path=ensure_project_relative(row["audio"]),
                    target_text=row["tgt_text"].strip(),
                )
            )
    return rows


def load_waveform(path: Path, sample_rate: int) -> torch.Tensor:
    """
    Charger une forme d'onde mono flottante au taux d'échantillonnage attendu.

    Paramètres :
        path : Fichier WAV (16 kHz après l'étape prepare).
        sample_rate : Hz attendus (défaut PRD 16000).

    Retour :
        1-D float tensor of shape ``[num_samples]``.

    Lève :
        ValueError: If the file sample rate does not match.
    """
    data, sr = sf.read(path.as_posix(), dtype="float32", always_2d=True)
    if sr != sample_rate:
        raise ValueError(
            f"Unexpected sample rate for {path}: {sr} (expected {sample_rate})"
        )
    # Mixer multi-canal en mono par moyenne des canaux.
    mono = data.mean(axis=1)
    return torch.from_numpy(mono)


def load_sentencepiece(path: Path) -> spm.SentencePieceProcessor:
    """
    Charger un modèle SentencePiece entraîné à l'étape ``3_spm``.

    Paramètres :
        path: ``.model`` file path.

    Retour :
        Loaded processor with BOS/EOS/PAD ids.

    Lève :
        RuntimeError: If the file cannot be loaded.
    """
    model = spm.SentencePieceProcessor()
    loaded = model.load(path.as_posix())
    if not loaded:
        raise RuntimeError(f"Could not load SentencePiece model: {path}")
    return model


class S3TModel(nn.Module):
    """
    Traduction vocale bout en bout : encodeur parole Hugging Face + décodeur Transformer.

    L'encodeur (Pantagruel) mappe l'audio brut français vers une séquence de vecteurs
    cachés (``memory``). Le décodeur prédit des sous-mots anglais avec auto-attention
    causale sur le préfixe cible et attention croisée vers ``memory``.
    """

    def __init__(
        self,
        *,
        encoder_name: str,
        vocab_size: int,
        hidden_dim: int,
        decoder_layers: int,
        decoder_heads: int,
        dropout: float,
        pad_id: int,
        max_positions: int = 512,
    ) -> None:
        """
        Construire les piles encodeur et décodeur.

        Paramètres :
            encoder_name : Identifiant modèle Hugging Face (ex. Pantagruel-Base).
            vocab_size : Taille du vocabulaire SentencePiece.
            hidden_dim : Largeur décodeur ; sorties encodeur projetées à cette taille.
            decoder_layers : Nombre de couches décodeur Transformer (PRD : 6).
            decoder_heads : Têtes d'attention dans chaque couche décodeur.
            dropout : Probabilité de dropout dans les couches décodeur.
            pad_id : Id token de padding pour masques et embeddings.
            max_positions : Longueur max séquence cible pour embeddings positionnels.
        """
        super().__init__()
        self.pad_id = pad_id
        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = int(self.encoder.config.hidden_size)
        self.encoder_proj: nn.Module
        if encoder_dim == hidden_dim:
            self.encoder_proj = nn.Identity()
        else:
            self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        self.tok_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_positions, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=decoder_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=decoder_layers)
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def freeze_encoder(self, should_freeze: bool) -> None:
        """
        Activer ou couper le flux de gradient dans l'encodeur Pantagruel (plan de gel RF-11).

        Paramètres :
            should_freeze : Si True, les poids encodeur ne sont pas mis à jour à cette étape.
        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not should_freeze

    def encode(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encoder des formes d'onde paddées en mémoire décodeur (clés/valeurs d'attention croisée).

        Paramètres :
            input_values : Tenseur flottant ``[batch, max_wave_len]``.
            attention_mask : Tenseur long, 1 pour échantillons réels et 0 pour padding.

        Retour :
            Projected hidden states ``[batch, time, hidden_dim]``.
        """
        encoded = self.encoder(
            input_values=input_values,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.encoder_proj(encoded)

    def decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        Exécuter le décodeur Transformer sur un préfixe de tokens cible.

        Paramètres :
            memory : Sortie encodeur ``[batch, enc_time, hidden_dim]``.
            tokens : Ids préfixe cible ``[batch, tgt_len]`` (BOS + tokens précédents).

        Retour :
            Logits ``[batch, tgt_len, vocab_size]`` for next-token prediction at each position.
        """
        seq_len = tokens.size(1)
        if seq_len > self.pos_embed.num_embeddings:
            raise ValueError(
                f"Token length {seq_len} exceeds max positions "
                f"{self.pos_embed.num_embeddings}"
            )
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        target = self.tok_embed(tokens) + self.pos_embed(positions)
        # Masque causal : la position i ne doit pas voir les positions > i (autoregressif).
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        target_padding = tokens.eq(self.pad_id)
        decoded = self.decoder(
            tgt=target,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=target_padding,
        )
        return self.output_proj(decoded)

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Passe avant complète pour l'entraînement : encoder l'audio, décoder le préfixe de tokens.

        Paramètres :
            input_values : Formes d'onde paddées.
            attention_mask : Masque de padding des formes d'onde.
            tokens : Tokens d'entrée en teacher forcing (``tokens_in`` du collate).

        Retour :
            Logits aligned with ``tokens_out`` for cross-entropy loss.
        """
        memory = self.encode(input_values, attention_mask)
        return self.decode(memory, tokens)


def collate_for_training(
    batch: list[ManifestSample],
    *,
    sp_model: spm.SentencePieceProcessor,
    sample_rate: int,
    max_target_tokens: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> dict[str, torch.Tensor]:
    """
    Regrouper des échantillons de manifest en lot train/éval avec teacher forcing.

    Construit des paires cibles décalées : ``tokens_in`` = BOS + target, ``tokens_out`` = target + EOS.
    Audio is padded to the longest waveform in the batch.

    Paramètres :
        batch : Liste d'échantillons manifest d'un lot DataLoader.
        sp_model : Processeur SentencePiece pour les cibles anglaises.
        sample_rate : Taux d'échantillonnage audio attendu.
        max_target_tokens : Tronquer les cibles encodées à cette longueur.
        pad_id, bos_id, eos_id: Special token ids from SPM.

    Retour :
        Dict with ``input_values``, ``attention_mask``, ``tokens_in``, ``tokens_out``.
    """
    waves = [load_waveform(sample.audio_path, sample_rate) for sample in batch]
    lengths = [wave.numel() for wave in waves]
    max_wave_len = max(lengths)
    input_values = torch.zeros((len(batch), max_wave_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(batch), max_wave_len), dtype=torch.long)
    for idx, wave in enumerate(waves):
        wave_len = wave.numel()
        input_values[idx, :wave_len] = wave
        attention_mask[idx, :wave_len] = 1

    encoded_targets: list[list[int]] = []
    for sample in batch:
        piece_ids = sp_model.encode(sample.target_text, out_type=int)[
            :max_target_tokens
        ]
        encoded_targets.append(piece_ids)

    # Teacher forcing : entrée décodeur = préfixe or ; cibles de perte décalées d'un pas.
    target_in: list[list[int]] = [[bos_id, *ids] for ids in encoded_targets]
    target_out: list[list[int]] = [[*ids, eos_id] for ids in encoded_targets]
    max_tgt = max(len(tokens) for tokens in target_in)

    tokens_in = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    tokens_out = torch.full((len(batch), max_tgt), pad_id, dtype=torch.long)
    for idx, (tin, tout) in enumerate(zip(target_in, target_out, strict=True)):
        tokens_in[idx, : len(tin)] = torch.tensor(tin, dtype=torch.long)
        tokens_out[idx, : len(tout)] = torch.tensor(tout, dtype=torch.long)

    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
    }


def greedy_decode_batch(
    *,
    model: S3TModel,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    Décodage glouton autorégressif pour un lot (pas de beam search).

    Encode l'audio une fois, puis ajoute répétitivement le token argmax à la dernière position
    until EOS or ``max_new_tokens`` is reached.

    Paramètres :
        model : ``S3TModel`` entraîné.
        input_values : Lot de formes d'onde.
        attention_mask : Masque des formes d'onde.
        bos_id, eos_id, pad_id: SPM special tokens.
        max_new_tokens : Nombre max d'étapes de génération après BOS.

    Retour :
        Token ids ``[batch, gen_len]`` including BOS and generated tokens.
    """
    model.eval()
    with torch.no_grad():
        memory = model.encode(input_values, attention_mask)
        batch_size = input_values.size(0)
        generated = torch.full(
            (batch_size, 1),
            fill_value=bos_id,
            dtype=torch.long,
            device=input_values.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_values.device)
        for _ in range(max_new_tokens):
            logits = model.decode(memory, generated)
            # Seule la dernière position prédit le token suivant.
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            # Continuer pad pour les séquences ayant déjà atteint EOS.
            next_token = torch.where(
                finished, torch.full_like(next_token, pad_id), next_token
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | next_token.eq(eos_id)
            if bool(torch.all(finished)):
                break
    return generated


def decode_ids_to_text(
    token_ids: list[int],
    *,
    sp_model: spm.SentencePieceProcessor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> str:
    """
    Reconvertir des ids de tokens générés ou de référence en chaîne détokenisée.

    Paramètres :
        token_ids : Liste plate d'ids SPM (peut inclure BOS/EOS/PAD).
        sp_model : Processeur SentencePiece.
        bos_id, eos_id, pad_id: Special tokens to strip.

    Retour :
        Detokenized English text, or empty string if no content tokens remain.
    """
    kept: list[int] = []
    for token in token_ids:
        if token in (bos_id, pad_id):
            continue
        if token == eos_id:
            break
        kept.append(token)
    if not kept:
        return ""
    return sp_model.decode(kept)


def resolve_run_dir(
    config: dict[str, Any],
    *,
    run_id: str,
    output_dir_override: Path | None,
) -> Path:
    """
    Résoudre le répertoire d'artefacts pour un run train/éval.

    Priority: CLI override > ``experiment.output_dir`` in config > default
    ``runs/<lang_pair>/<run_id>``.

    Paramètres :
        config : Dict YAML d'expérience.
        run_id : Identifiant unique du run.
        output_dir_override : Répertoire explicite optionnel depuis la CLI.

    Retour :
        Absolute path to the run folder.
    """
    if output_dir_override is not None:
        return output_dir_override
    from_config = deep_get(config, "experiment.output_dir", None)
    if from_config:
        base = ensure_project_relative(from_config)
        return base.parent / run_id if base.name != run_id else base
    langpair = str(deep_get(config, "experiment.lang_pair", "fr-en"))
    return PROJECT_ROOT / "runs" / langpair / run_id


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """
    Écrire un artefact JSON en créant les répertoires parents si nécessaire.

    Paramètres :
        path : Chemin du fichier de sortie.
        payload : Dict sérialisable (métriques, rapports, etc.).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
