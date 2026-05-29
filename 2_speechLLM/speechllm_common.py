#!/usr/bin/env python3
"""
Bibliothèque partagée pour la pipeline speechLLM (train, evaluate, infer).

Architecture B1 : encodeur Pantagruel (gelé) → downsampling temporel → projecteur
(entraînable) → embeddings injectés dans un LLM causal (gelé) pour générer l'anglais.

Entrées typiques : config YAML, manifests TSV m-TEDx, checkpoint projecteur.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from scripts_communs.st_common import (
    ManifestSample,
    deep_get,
    ensure_project_relative,
    load_waveform,
    load_yaml_config,
    read_manifest,
    resolve_run_dir,
    set_seed,
    write_json,
)
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

SPEECHLLM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SPEECHLLM_ROOT.parent

IGNORE_INDEX = -100


@dataclass
class EncodedBatch:
    """Lot prêt pour ``SpeechLLMModel.forward_train``."""

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def downsample_encoder_states(
    hidden: torch.Tensor,
    encoder_mask: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concaténer ``k`` frames consécutives sur la dimension des features (SLAM-ASR).

    Paramètres :
        hidden : États encodeur ``[batch, time, dim]``.
        encoder_mask : Masque 1/0 ``[batch, time]``.
        k : Facteur de downsampling (>= 1).

    Retour :
        Tenseurs ``[batch, time//k, dim*k]`` et masque associé.
    """
    if k <= 1:
        return hidden, encoder_mask
    batch_size, time_steps, dim = hidden.shape
    trimmed = time_steps - (time_steps % k)
    if trimmed == 0:
        return hidden.new_zeros((batch_size, 0, dim * k)), encoder_mask.new_zeros(
            (batch_size, 0)
        )
    trimmed_hidden = hidden[:, :trimmed, :]
    trimmed_mask = encoder_mask[:, :trimmed]
    new_time = trimmed // k
    reshaped = trimmed_hidden.reshape(batch_size, new_time, k * dim)
    mask_blocks = trimmed_mask.reshape(batch_size, new_time, k)
    new_mask = mask_blocks.amax(dim=-1).long()
    return reshaped, new_mask


class SpeechLLMModel(nn.Module):
    """
    Pantagruel (parole) + projecteur + LLM causal pour ST fr→en.

    Seul le projecteur est entraîné en B1 ; encodeur et LLM restent gelés.
    """

    def __init__(
        self,
        *,
        encoder_name: str,
        llm_name: str,
        downsample_k: int = 5,
        projector_hidden: int = 2048,
        freeze_encoder: bool = True,
        freeze_llm: bool = True,
    ) -> None:
        """
        Initialiser le modèle speechLLM (encodeur + projecteur + LLM).

        Paramètres :
            encoder_name : Identifiant Hugging Face de l'encodeur parole (Pantagruel).
            llm_name : Identifiant Hugging Face du LLM causal.
            downsample_k : Facteur de downsampling temporel (concaténation k-frames).
            projector_hidden : Dimension cachée du MLP projecteur.
            freeze_encoder : Geler les poids de l'encodeur.
            freeze_llm : Geler les poids du LLM.
        """
        super().__init__()
        self.downsample_k = max(1, int(downsample_k))
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        encoder_dim = int(self.encoder.config.hidden_size)
        llm_dim = int(self.llm.config.hidden_size)
        self.encoder_proj = (
            nn.Identity() if encoder_dim == llm_dim else nn.Linear(encoder_dim, llm_dim)
        )
        input_dim = encoder_dim * self.downsample_k
        self.projector = nn.Sequential(
            nn.Linear(input_dim, projector_hidden),
            nn.ReLU(),
            nn.Linear(projector_hidden, llm_dim),
        )

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        if freeze_llm:
            for parameter in self.llm.parameters():
                parameter.requires_grad = False
            self.llm.eval()

    @property
    def llm_hidden_size(self) -> int:
        """Dimension des embeddings attendus par le LLM."""
        return int(self.llm.config.hidden_size)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Paramètres mis à jour par l'optimiseur (projecteur + éventuelle projection encodeur)."""
        params: list[nn.Parameter] = list(self.projector.parameters())
        if isinstance(self.encoder_proj, nn.Linear):
            params.extend(self.encoder_proj.parameters())
        return params

    def encode_speech(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encoder l'audio et projeter vers l'espace embedding du LLM.

        Retour :
            ``speech_embeds`` ``[batch, speech_len, llm_dim]``, masque speech.
        """
        with torch.set_grad_enabled(
            any(parameter.requires_grad for parameter in self.encoder.parameters())
        ):
            encoded = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            ).last_hidden_state
        projected = self.encoder_proj(encoded)
        downsampled, speech_mask = downsample_encoder_states(
            projected,
            attention_mask,
            k=self.downsample_k,
        )
        speech_embeds = self.projector(downsampled)
        return speech_embeds, speech_mask

    def _embed_text_ids(
        self, token_ids: list[int], device: torch.device
    ) -> torch.Tensor:
        """
        Convertir des ids token en embeddings d'entrée du LLM.

        Utile pour concaténer (1) du texte sérialisé (`USER:`, prompt, etc.) et
        (2) des embeddings speech projetés, via `inputs_embeds`.
        """
        tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        return self.llm.get_input_embeddings()(tensor)

    def build_sequence(
        self,
        *,
        speech_embeds: torch.Tensor,
        target_text: str,
        prompt: str,
        max_target_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Construire ``inputs_embeds`` et ``labels`` pour un échantillon.

        Format : ``USER:`` + speech + prompt + ``ASSISTANT:`` + cible (loss sur la cible).
        """
        user_ids = self.tokenizer.encode("USER: ", add_special_tokens=False)
        mid_ids = self.tokenizer.encode(
            f"{prompt} ASSISTANT: ",
            add_special_tokens=False,
        )
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)[
            :max_target_tokens
        ]

        chunks: list[torch.Tensor] = []
        labels: list[int] = []

        if user_ids:
            chunks.append(self._embed_text_ids(user_ids, device))
            labels.extend([IGNORE_INDEX] * len(user_ids))

        chunks.append(speech_embeds)
        labels.extend([IGNORE_INDEX] * speech_embeds.size(0))

        if mid_ids:
            chunks.append(self._embed_text_ids(mid_ids, device))
            labels.extend([IGNORE_INDEX] * len(mid_ids))

        if target_ids:
            chunks.append(self._embed_text_ids(target_ids, device))
            labels.extend(target_ids)

        inputs_embeds = torch.cat(chunks, dim=0)
        label_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        return inputs_embeds, label_tensor

    def forward_train(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        target_texts: list[str],
        *,
        prompt: str,
        max_target_tokens: int,
    ) -> torch.Tensor:
        """
        Calculer la loss language-modeling sur les tokens de traduction uniquement.
        """
        speech_embeds, _ = self.encode_speech(input_values, attention_mask)
        device = input_values.device
        batch_size = input_values.size(0)

        sequences: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        for index in range(batch_size):
            sample_embeds, sample_labels = self.build_sequence(
                speech_embeds=speech_embeds[index],
                target_text=target_texts[index],
                prompt=prompt,
                max_target_tokens=max_target_tokens,
                device=device,
            )
            sequences.append(sample_embeds)
            label_rows.append(sample_labels)

        max_len = max(row.size(0) for row in sequences)
        hidden = self.llm_hidden_size
        padded_embeds = torch.zeros(
            (batch_size, max_len, hidden),
            dtype=sequences[0].dtype,
            device=device,
        )
        padded_labels = torch.full(
            (batch_size, max_len),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
            device=device,
        )
        padded_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )
        for index, (embeds, labs) in enumerate(zip(sequences, label_rows, strict=True)):
            length = embeds.size(0)
            padded_embeds[index, :length] = embeds
            padded_labels[index, :length] = labs
            padded_mask[index, :length] = 1

        outputs = self.llm(
            inputs_embeds=padded_embeds,
            attention_mask=padded_mask,
            labels=padded_labels,
        )
        return outputs.loss

    def build_prompt_embeds(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeddings prefixe ``USER:`` + speech + ``prompt ASSISTANT:`` (sans cible)."""
        speech_embeds, speech_mask = self.encode_speech(input_values, attention_mask)
        device = input_values.device
        batch_size = input_values.size(0)
        user_ids = self.tokenizer.encode("USER: ", add_special_tokens=False)
        mid_ids = self.tokenizer.encode(
            f"{prompt} ASSISTANT: ",
            add_special_tokens=False,
        )

        sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for index in range(batch_size):
            chunks: list[torch.Tensor] = []
            mask_parts: list[int] = []
            if user_ids:
                chunks.append(self._embed_text_ids(user_ids, device))
                mask_parts.extend([1] * len(user_ids))
            speech = speech_embeds[index]
            valid = int(speech_mask[index].sum().item())
            chunks.append(speech[:valid])
            mask_parts.extend([1] * valid)
            if mid_ids:
                chunks.append(self._embed_text_ids(mid_ids, device))
                mask_parts.extend([1] * len(mid_ids))
            seq = torch.cat(chunks, dim=0)
            sequences.append(seq)
            masks.append(torch.tensor(mask_parts, dtype=torch.long, device=device))

        max_len = max(seq.size(0) for seq in sequences)
        hidden = self.llm_hidden_size
        padded = torch.zeros(
            (batch_size, max_len, hidden),
            dtype=sequences[0].dtype,
            device=device,
        )
        attn = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        for index, (seq, mask_row) in enumerate(zip(sequences, masks, strict=True)):
            length = seq.size(0)
            padded[index, :length] = seq
            attn[index, :length] = mask_row
        return padded, attn

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt: str,
        max_new_tokens: int,
        num_beams: int = 1,
    ) -> list[str]:
        """Générer la traduction anglaise pour un lot d'audios."""
        self.llm.eval()
        inputs_embeds, attn = self.build_prompt_embeds(
            input_values,
            attention_mask,
            prompt=prompt,
        )
        prompt_lengths = attn.sum(dim=1)
        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=max(1, num_beams),
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        results: list[str] = []
        for row_idx in range(generated.size(0)):
            token_ids = generated[row_idx].tolist()
            # Si la sortie ré-inclut le préfixe, ne garder que les tokens générés.
            prefix_len = int(prompt_lengths[row_idx].item())
            if len(token_ids) > prefix_len:
                token_ids = token_ids[prefix_len:]
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            if not text:
                full = self.tokenizer.decode(
                    generated[row_idx], skip_special_tokens=True
                ).strip()
                marker = "ASSISTANT:"
                text = full.split(marker, 1)[-1].strip() if marker in full else full
            results.append(text)
        return results


def collate_speechllm_batch(
    batch: list[ManifestSample],
    *,
    sample_rate: int,
) -> dict[str, Any]:
    """Regrouper audio paddé et textes cibles bruts (pas de SPM)."""
    waves = [load_waveform(sample.audio_path, sample_rate) for sample in batch]
    lengths = [wave.numel() for wave in waves]
    max_wave_len = max(lengths) if lengths else 0
    input_values = torch.zeros((len(batch), max_wave_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(batch), max_wave_len), dtype=torch.long)
    for index, wave in enumerate(waves):
        wave_len = wave.numel()
        input_values[index, :wave_len] = wave
        attention_mask[index, :wave_len] = 1
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "target_texts": [sample.target_text for sample in batch],
    }


def load_speechllm_from_config(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> SpeechLLMModel:
    """Instancier ``SpeechLLMModel`` depuis une config YAML."""
    encoder_name = str(
        deep_get(config, "model.encoder_name", "PantagrueLLM/Pantagruel-Base")
    )
    llm_name = str(deep_get(config, "model.llm_name", "microsoft/phi-2"))
    downsample_k = int(deep_get(config, "model.downsample_k", 5))
    projector_hidden = int(deep_get(config, "model.projector_hidden", 2048))
    freeze_encoder = bool(deep_get(config, "model.freeze_encoder", True))
    freeze_llm = bool(deep_get(config, "model.freeze_llm", True))
    model = SpeechLLMModel(
        encoder_name=encoder_name,
        llm_name=llm_name,
        downsample_k=downsample_k,
        projector_hidden=projector_hidden,
        freeze_encoder=freeze_encoder,
        freeze_llm=freeze_llm,
    )
    return model.to(device)


def load_speechllm_checkpoint(path: Path) -> dict[str, Any]:
    """
    Charger un checkpoint projecteur écrit par ``train.py``.

    Lève :
        FileNotFoundError, ValueError : Fichier absent ou payload invalide.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "trainable_state" not in payload:
        raise ValueError(f"Invalid speechLLM checkpoint: {path}")
    return payload


def load_projector_checkpoint(
    model: SpeechLLMModel,
    payload: dict[str, Any],
) -> None:
    """Restaurer les poids entraînables (projecteur) depuis un checkpoint."""
    state = payload.get("trainable_state")
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)


def save_projector_checkpoint(
    *,
    path: Path,
    model: SpeechLLMModel,
    config: dict[str, Any],
    run_id: str,
    git_commit: str,
    update: int,
    best_bleu_dev: float,
) -> None:
    """Sauvegarder projecteur (+ métadonnées) pour evaluate/infer."""
    trainable = {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if key.startswith("projector.") or key.startswith("encoder_proj.")
    }
    torch.save(
        {
            "pipeline": "speechllm",
            "run_id": run_id,
            "config": config,
            "trainable_state": trainable,
            "git_commit": git_commit,
            "update": update,
            "best_bleu_dev": best_bleu_dev,
        },
        path,
    )


def resolve_speechllm_config_path(path: Path) -> Path:
    """Résoudre un chemin de config relatif à la racine projet."""
    return ensure_project_relative(path)


__all__ = [
    "EncodedBatch",
    "IGNORE_INDEX",
    "SpeechLLMModel",
    "collate_speechllm_batch",
    "downsample_encoder_states",
    "load_speechllm_checkpoint",
    "load_projector_checkpoint",
    "load_speechllm_from_config",
    "load_yaml_config",
    "read_manifest",
    "resolve_run_dir",
    "resolve_speechllm_config_path",
    "save_projector_checkpoint",
    "set_seed",
    "write_json",
]
