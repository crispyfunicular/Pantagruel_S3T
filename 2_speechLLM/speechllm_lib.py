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

# Formats de prompt supportés pour l'injection speech → LLM (B1 Phi-2, B2bis Qwen/Mistral).
SUPPORTED_PROMPT_FORMATS = frozenset({"phi2", "qwen_chatml", "mistral_inst"})


@dataclass(frozen=True)
class PromptTextParts:
    """Fragments textuels autour des embeddings parole pour un template chat donné."""

    prefix: str
    suffix: str
    assistant_marker: str


def resolve_prompt_format(config: dict[str, Any]) -> str:
    """
    Déterminer le format de prompt depuis la config YAML.

    Si ``prompt.format`` est absent, on infère depuis ``model.llm_name`` pour
    rétrocompatibilité B1 (Phi-2 → ``phi2``).
    """
    explicit = deep_get(config, "prompt.format")
    if explicit is not None:
        format_name = str(explicit).strip().lower()
        if format_name not in SUPPORTED_PROMPT_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_PROMPT_FORMATS))
            raise ValueError(
                f"Unsupported prompt.format={format_name!r} (expected: {supported})"
            )
        return format_name

    llm_name = str(deep_get(config, "model.llm_name", "microsoft/phi-2")).lower()
    if "qwen" in llm_name:
        return "qwen_chatml"
    if "mistral" in llm_name:
        return "mistral_inst"
    return "phi2"


def build_prompt_text_parts(format_name: str, prompt: str) -> PromptTextParts:
    """
    Construire les fragments textuels avant/après les embeddings parole.

    Séquence entraînement / inférence :
    ``prefix`` + speech_embeds + ``suffix`` + (cible en train uniquement).
    """
    if format_name == "phi2":
        return PromptTextParts(
            prefix="USER: ",
            suffix=f"{prompt} ASSISTANT: ",
            assistant_marker="ASSISTANT:",
        )
    if format_name == "qwen_chatml":
        return PromptTextParts(
            prefix="<|im_start|>user\n",
            suffix=f"{prompt}\n<|im_start|>assistant\n",
            assistant_marker="assistant",
        )
    if format_name == "mistral_inst":
        return PromptTextParts(
            prefix="[INST] ",
            suffix=f"{prompt} [/INST] ",
            assistant_marker="[/INST]",
        )
    supported = ", ".join(sorted(SUPPORTED_PROMPT_FORMATS))
    raise ValueError(
        f"Unsupported prompt format {format_name!r} (expected: {supported})"
    )


def _load_causal_lm(
    llm_name: str,
    *,
    load_in_4bit: bool,
    load_in_8bit: bool,
    trust_remote_code: bool,
    device: torch.device,
) -> AutoModelForCausalLM:
    """
    Charger un LLM causal Hugging Face, avec quantisation optionnelle (B2bis 7B).

    ``bitsandbytes`` est requis uniquement si ``load_in_4bit`` ou ``load_in_8bit``.
    """
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if load_in_4bit or load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "bitsandbytes est requis pour model.load_in_4bit / load_in_8bit "
                "(pip install bitsandbytes)"
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )
        kwargs["device_map"] = {"": str(device)}
    model = AutoModelForCausalLM.from_pretrained(llm_name, **kwargs)
    if not (load_in_4bit or load_in_8bit):
        model = model.to(device)
    return model


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
        trust_remote_code: bool = False,
        llm_trust_remote_code: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        prompt_format: str = "phi2",
        device: torch.device | None = None,
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
            trust_remote_code : ``trust_remote_code`` pour l'encodeur Pantagruel.
            llm_trust_remote_code : ``trust_remote_code`` pour le LLM / tokenizer.
            load_in_4bit : Charger le LLM en 4-bit (nécessite ``bitsandbytes``).
            load_in_8bit : Charger le LLM en 8-bit (nécessite ``bitsandbytes``).
            prompt_format : Template chat (`phi2`, `qwen_chatml`, `mistral_inst`).
            device : Périphérique cible (requis si quantisation LLM activée).
        """
        super().__init__()
        self.downsample_k = max(1, int(downsample_k))
        self.llm_name = llm_name
        format_name = str(prompt_format).strip().lower()
        if format_name not in SUPPORTED_PROMPT_FORMATS:
            supported = ", ".join(sorted(SUPPORTED_PROMPT_FORMATS))
            raise ValueError(
                f"Unsupported prompt_format={format_name!r} (expected: {supported})"
            )
        self.prompt_format = format_name
        self._quantized_llm = bool(load_in_4bit or load_in_8bit)
        if self._quantized_llm and device is None:
            raise ValueError("device is required when loading a quantized LLM")

        # Certains checkpoints Pantagruel exposent du code HF custom (trust_remote_code requis).
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            trust_remote_code=trust_remote_code,
        )
        llm_device = device if device is not None else torch.device("cpu")
        self.llm = _load_causal_lm(
            llm_name,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            trust_remote_code=llm_trust_remote_code,
            device=llm_device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            trust_remote_code=llm_trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        encoder_dim = _resolve_encoder_output_dim(self.encoder)
        llm_dim = int(self.llm.config.hidden_size)
        self.encoder_proj = (
            nn.Identity() if encoder_dim == llm_dim else nn.Linear(encoder_dim, llm_dim)
        )
        # Après encoder_proj + downsample k, la taille d'entrée du projecteur est llm_dim * k.
        input_dim = llm_dim * self.downsample_k
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
        """Paramètres mis à jour par l'optimiseur.

        Par défaut (B1), seul le projecteur est entraîné. Si l'encodeur n'est pas gelé,
        on inclut aussi ses paramètres afin que `freeze_encoder: false` ait un effet réel
        (sinon on paie le backward sans mettre à jour les poids).
        """
        params: list[nn.Parameter] = list(self.projector.parameters())
        if isinstance(self.encoder_proj, nn.Linear):
            params.extend(self.encoder_proj.parameters())
        # Encoder: entraîné uniquement si dégelé via `freeze_encoder: false`.
        if any(parameter.requires_grad for parameter in self.encoder.parameters()):
            params.extend([p for p in self.encoder.parameters() if p.requires_grad])
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

    def _prompt_token_ids(self, prompt: str) -> tuple[list[int], list[int]]:
        """Ids tokenizer pour le préfixe et le suffixe textuels autour des embeddings parole."""
        parts = build_prompt_text_parts(self.prompt_format, prompt)
        prefix_ids = self.tokenizer.encode(parts.prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(parts.suffix, add_special_tokens=False)
        return prefix_ids, suffix_ids

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

        Format : ``prefix`` + speech + ``suffix`` + cible (loss sur la cible uniquement).
        """
        prefix_ids, suffix_ids = self._prompt_token_ids(prompt)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)[
            :max_target_tokens
        ]

        chunks: list[torch.Tensor] = []
        labels: list[int] = []

        if prefix_ids:
            chunks.append(self._embed_text_ids(prefix_ids, device))
            labels.extend([IGNORE_INDEX] * len(prefix_ids))

        chunks.append(speech_embeds)
        labels.extend([IGNORE_INDEX] * speech_embeds.size(0))

        if suffix_ids:
            chunks.append(self._embed_text_ids(suffix_ids, device))
            labels.extend([IGNORE_INDEX] * len(suffix_ids))

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
        """Embeddings préfixe chat + speech + suffixe (sans cible)."""
        speech_embeds, speech_mask = self.encode_speech(input_values, attention_mask)
        device = input_values.device
        batch_size = input_values.size(0)
        prefix_ids, suffix_ids = self._prompt_token_ids(prompt)

        sequences: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for index in range(batch_size):
            chunks: list[torch.Tensor] = []
            mask_parts: list[int] = []
            if prefix_ids:
                chunks.append(self._embed_text_ids(prefix_ids, device))
                mask_parts.extend([1] * len(prefix_ids))
            speech = speech_embeds[index]
            valid = int(speech_mask[index].sum().item())
            chunks.append(speech[:valid])
            mask_parts.extend([1] * valid)
            if suffix_ids:
                chunks.append(self._embed_text_ids(suffix_ids, device))
                mask_parts.extend([1] * len(suffix_ids))
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
        # Aligner le dtype sur le LLM (évite Half vs Float en beam search hors autocast).
        llm_dtype = next(self.llm.parameters()).dtype
        inputs_embeds = inputs_embeds.to(dtype=llm_dtype)
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
                marker = build_prompt_text_parts(
                    self.prompt_format,
                    prompt,
                ).assistant_marker
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
    trust_remote_code = bool(deep_get(config, "model.trust_remote_code", False))
    llm_trust_remote_code = bool(
        deep_get(config, "model.llm_trust_remote_code", trust_remote_code)
    )
    load_in_4bit = bool(deep_get(config, "model.load_in_4bit", False))
    load_in_8bit = bool(deep_get(config, "model.load_in_8bit", False))
    prompt_format = resolve_prompt_format(config)
    model = SpeechLLMModel(
        encoder_name=encoder_name,
        llm_name=llm_name,
        downsample_k=downsample_k,
        projector_hidden=projector_hidden,
        freeze_encoder=freeze_encoder,
        freeze_llm=freeze_llm,
        trust_remote_code=trust_remote_code,
        llm_trust_remote_code=llm_trust_remote_code,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        prompt_format=prompt_format,
        device=device,
    )
    if model._quantized_llm:
        # Le LLM quantifié est déjà placé via device_map ; encoder + projecteur sur GPU.
        model.encoder = model.encoder.to(device)
        model.encoder_proj = model.encoder_proj.to(device)
        model.projector = model.projector.to(device)
        return model
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
    """Restaurer projecteur (et encodeur si présent dans le payload) depuis un checkpoint."""
    state = payload.get("trainable_state")
    if isinstance(state, dict):
        model.load_state_dict(state, strict=False)


def _speechllm_checkpoint_prefixes(config: dict[str, Any]) -> tuple[str, ...]:
    """
    Préfixes des tenseurs à persister pour evaluate/infer.

    B1 gelé : projecteur (+ ``encoder_proj`` si présent). Si ``freeze_encoder: false``,
    inclure aussi l'encodeur fine-tuné — sinon l'éval recharge le HF de base et le BLEU
    s'effondre.
    """
    prefixes: list[str] = ["projector.", "encoder_proj."]
    if not bool(deep_get(config, "model.freeze_encoder", True)):
        prefixes.append("encoder.")
    return tuple(prefixes)


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
    """Sauvegarder les poids entraînés (projecteur, encodeur si dégelé) pour evaluate/infer."""
    prefixes = _speechllm_checkpoint_prefixes(config)
    trainable = {
        key: value.cpu()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
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
    "PromptTextParts",
    "SUPPORTED_PROMPT_FORMATS",
    "SpeechLLMModel",
    "build_prompt_text_parts",
    "collate_speechllm_batch",
    "downsample_encoder_states",
    "load_speechllm_checkpoint",
    "load_projector_checkpoint",
    "load_speechllm_from_config",
    "load_yaml_config",
    "read_manifest",
    "resolve_prompt_format",
    "resolve_run_dir",
    "resolve_speechllm_config_path",
    "save_projector_checkpoint",
    "set_seed",
    "write_json",
]
