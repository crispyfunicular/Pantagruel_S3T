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
from transformers import AutoModel

from scripts_communs.config_utils import deep_get, load_yaml_config, write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

__all__ = [
    "ManifestSample",
    "PROJECT_ROOT",
    "S3TModel",
    "beam_decode_batch",
    "decode_batch",
    "deep_get",
    "ensure_project_relative",
    "greedy_decode_batch",
    "SpecAugmentConfig",
    "apply_feature_freq_mask",
    "apply_waveform_time_mask",
    "load_waveform",
    "load_yaml_config",
    "parse_spec_augment_config",
    "read_manifest",
    "set_seed",
    "write_json",
]


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


@dataclass(frozen=True)
class SpecAugmentConfig:
    """Paramètres SpecAugment (masquage temporel + fréquentiel, entraînement ST)."""

    enabled: bool = False
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_freq_prob: float = 0.0
    mask_freq_length: int = 27


def parse_spec_augment_config(config: dict[str, Any]) -> SpecAugmentConfig:
    """
    Lire la section ``spec_augment`` d'une config YAML ST.

    ``mask_time_length`` compte des fenêtres d'environ 10 ms à ``sample_rate`` Hz
    (aligné LeBenchmark / fairseq : masques courts sur la timeline acoustique).
    """
    block = deep_get(config, "spec_augment", {}) or {}
    if not isinstance(block, dict):
        block = {}
    return SpecAugmentConfig(
        enabled=bool(block.get("enabled", False)),
        mask_time_prob=float(block.get("mask_time_prob", 0.05)),
        mask_time_length=int(block.get("mask_time_length", 10)),
        mask_freq_prob=float(block.get("mask_freq_prob", 0.0)),
        mask_freq_length=int(block.get("mask_freq_length", 27)),
    )


def apply_waveform_time_mask(
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    spec_augment: SpecAugmentConfig,
    sample_rate: int,
) -> torch.Tensor:
    """
    Appliquer un masquage temporel aléatoire (SpecAugment) sur des formes d'onde paddées.

    Réservé à l'entraînement : zéro des échantillons sur un segment contigu par utterance
    tirée au sort selon ``mask_time_prob``.
    """
    if not spec_augment.enabled or spec_augment.mask_time_prob <= 0:
        return input_values
    mask_samples = max(1, spec_augment.mask_time_length * sample_rate // 100)
    out = input_values.clone()
    batch_size = out.size(0)
    for batch_idx in range(batch_size):
        if random.random() > spec_augment.mask_time_prob:
            continue
        valid_len = int(attention_mask[batch_idx].sum().item())
        if valid_len <= mask_samples:
            continue
        start = random.randint(0, valid_len - mask_samples)
        out[batch_idx, start : start + mask_samples] = 0.0
    return out


def apply_feature_freq_mask(
    features: torch.Tensor,
    *,
    spec_augment: SpecAugmentConfig,
) -> torch.Tensor:
    """
    Appliquer un masquage « fréquentiel » sur les features encodeur ``[batch, time, hidden]``.

    Sur les tenseurs cachés Pantagruel, on masque une bande contiguë de dimensions de
    feature (analogue aux bandes de mélatre SpecAugment fairseq).
    """
    if not spec_augment.enabled or spec_augment.mask_freq_prob <= 0:
        return features
    mask_width = max(1, spec_augment.mask_freq_length)
    hidden_dim = features.size(-1)
    if mask_width >= hidden_dim:
        return features
    out = features.clone()
    batch_size = out.size(0)
    for batch_idx in range(batch_size):
        if random.random() > spec_augment.mask_freq_prob:
            continue
        start = random.randint(0, hidden_dim - mask_width)
        out[batch_idx, :, start : start + mask_width] = 0.0
    return out


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


def build_s3t_model(
    config: dict[str, Any],
    *,
    vocab_size: int,
    pad_id: int,
    max_positions: int,
) -> S3TModel:
    """
    Instancier ``S3TModel`` à partir d'une config YAML (étapes 4–6 / variante 5).

    Paramètres :
        config : Config d'expérience (clés ``model.*``).
        vocab_size : Taille du vocabulaire SentencePiece.
        pad_id : Id token PAD SPM.
        max_positions : Longueur max positions décodeur.

    Retour :
        Modèle prêt pour ``.to(device)``.
    """
    return S3TModel(
        encoder_name=str(
            deep_get(config, "model.encoder_name", "PantagrueLLM/Pantagruel-Base")
        ),
        vocab_size=vocab_size,
        hidden_dim=int(deep_get(config, "model.hidden_dim", 768)),
        decoder_layers=int(deep_get(config, "model.decoder_layers", 6)),
        decoder_heads=int(deep_get(config, "model.decoder_heads", 8)),
        dropout=float(deep_get(config, "model.dropout", 0.1)),
        pad_id=pad_id,
        max_positions=max_positions,
        trust_remote_code=bool(deep_get(config, "model.trust_remote_code", False)),
        encoder_api=str(deep_get(config, "model.encoder_api", "default")),
    )


def _resolve_encoder_output_dim(encoder: nn.Module) -> int:
    """
    Déterminer la dimension de sortie réelle de l'encodeur Pantagruel.

    Sur les checkpoints Large (14k / 114k), ``config.hidden_size`` peut indiquer 768
    alors que ``last_hidden_state`` est en 1024 ; on sonde un forward minimal.
    """
    config_dim = int(encoder.config.hidden_size)
    was_training = encoder.training
    encoder.eval()
    with torch.no_grad():
        probe_wav = torch.randn(1, 8000)
        probe_mask = torch.ones(1, probe_wav.size(1), dtype=torch.long)
        outputs = encoder(input_values=probe_wav, attention_mask=probe_mask)
        actual_dim = int(outputs.last_hidden_state.shape[-1])
    if was_training:
        encoder.train()
    return actual_dim if actual_dim != config_dim else config_dim


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
        trust_remote_code: bool = False,
        encoder_api: str = "default",
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
            trust_remote_code : Requis pour les checkpoints Pantagruel ``speech_text``.
            encoder_api : ``default`` (``last_hidden_state``) ou ``speech_text`` (``mode=AUDIO``).
        """
        super().__init__()
        self.pad_id = pad_id
        self.encoder_api = encoder_api
        load_kwargs: dict[str, Any] = {}
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True
        self.encoder = AutoModel.from_pretrained(encoder_name, **load_kwargs)
        encoder_dim = _resolve_encoder_output_dim(self.encoder)
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
        if self.encoder_api == "speech_text":
            outputs = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
                mode="AUDIO",
            )
            hidden = outputs.audio_output.last_hidden_state
        else:
            hidden = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            ).last_hidden_state
        return self.encoder_proj(hidden)

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


def _beam_decode_single(
    *,
    model: S3TModel,
    memory: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    beam_size: int,
) -> torch.Tensor:
    """
    Beam search sur un seul échantillon (mémoire encodeur ``[1, enc_time, dim]``).

    Conserve les ``beam_size`` meilleures hypothèses par log-probabilité cumulée ;
    les séquences terminées par EOS ne sont plus étendues mais restent candidates.
    """
    device = memory.device
    beams: list[tuple[list[int], float, bool]] = [([bos_id], 0.0, False)]

    for _ in range(max_new_tokens):
        candidates: list[tuple[list[int], float, bool]] = []
        active = [beam for beam in beams if not beam[2]]
        finished = [beam for beam in beams if beam[2]]
        candidates.extend(finished)

        if not active:
            break

        for tokens, log_prob, _ in active:
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decode(memory, token_tensor)[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
            top_log_probs, top_ids = torch.topk(
                log_probs, min(beam_size, log_probs.numel())
            )
            for token_log_prob, token_id in zip(
                top_log_probs.tolist(),
                top_ids.tolist(),
                strict=True,
            ):
                new_tokens = [*tokens, token_id]
                new_log_prob = log_prob + token_log_prob
                is_finished = token_id == eos_id
                candidates.append((new_tokens, new_log_prob, is_finished))

        candidates.sort(key=lambda item: item[1], reverse=True)
        beams = candidates[:beam_size]
        if beams and all(finished for _, _, finished in beams):
            break

    best_tokens = max(beams, key=lambda item: item[1])[0]
    return torch.tensor(best_tokens, dtype=torch.long, device=device)


def beam_decode_batch(
    *,
    model: S3TModel,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    beam_size: int,
) -> torch.Tensor:
    """
    Décodage par faisceau (beam search) pour un lot — un faisceau indépendant par ligne.

    Paramètres :
        beam_size : Largeur du faisceau (``<= 1`` délègue au greedy).

    Retour :
        Token ids ``[batch, gen_len]`` paddés à droite avec ``pad_id`` si longueurs variables.
    """
    if beam_size <= 1:
        return greedy_decode_batch(
            model=model,
            input_values=input_values,
            attention_mask=attention_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=max_new_tokens,
        )

    model.eval()
    with torch.no_grad():
        memory = model.encode(input_values, attention_mask)
        batch_size = input_values.size(0)
        sequences: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            seq = _beam_decode_single(
                model=model,
                memory=memory[batch_idx : batch_idx + 1],
                bos_id=bos_id,
                eos_id=eos_id,
                pad_id=pad_id,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
            )
            sequences.append(seq.cpu())

        max_len = max(seq.size(0) for seq in sequences)
        output = torch.full(
            (batch_size, max_len),
            fill_value=pad_id,
            dtype=torch.long,
        )
        for batch_idx, seq in enumerate(sequences):
            output[batch_idx, : seq.size(0)] = seq
        return output


def decode_batch(
    *,
    model: S3TModel,
    input_values: torch.Tensor,
    attention_mask: torch.Tensor,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int,
    beam_size: int = 1,
) -> torch.Tensor:
    """
    Point d'entrée unifié : greedy si ``beam_size <= 1``, sinon beam search.

    Paramètres :
        beam_size : Largeur du faisceau (objectif papier Pantagruel : 5).
    """
    if beam_size <= 1:
        return greedy_decode_batch(
            model=model,
            input_values=input_values,
            attention_mask=attention_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=max_new_tokens,
        )
    return beam_decode_batch(
        model=model,
        input_values=input_values,
        attention_mask=attention_mask,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
        beam_size=beam_size,
    )


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
