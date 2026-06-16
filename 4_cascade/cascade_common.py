#!/usr/bin/env python3
"""
Bibliothèque partagée — baseline cascade ASR→MT (fr → en).

Rôle dans le pipeline S3T :
- charger la configuration YAML (`4_cascade/configs/...`) ;
- enchaîner ASR (audio FR → texte FR) puis MT (texte FR → texte EN) ;
- exposer des fonctions réutilisables par `evaluate_cascade.py` et `infer_cascade.py`.

Backends implémentés :
- ASR ``whisper`` : Hugging Face ``WhisperForConditionalGeneration`` (langue FR, tâche transcribe).
- MT ``marian`` : Hugging Face ``MarianMTModel`` (ex. ``Helsinki-NLP/opus-mt-fr-en``).

Les modèles sont chargés une fois par couple (backend, model_id) et réutilisés sur tout un run evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from scripts_communs.st_common import load_waveform
from speechLLM.speechllm_lib import PROJECT_ROOT, deep_get

CASCADE_ROOT = Path(__file__).resolve().parent

# Codes de sortie documentés pour les scripts Cascade (alignés sur scripts/ PRD).
EXIT_SUCCESS = 0
EXIT_CONFIG = 2
EXIT_NOT_IMPLEMENTED = 3

DEFAULT_ASR_BACKEND = "whisper"
DEFAULT_ASR_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_MT_BACKEND = "marian"
DEFAULT_MT_MODEL_ID = "Helsinki-NLP/opus-mt-fr-en"
DEFAULT_ASR_LANGUAGE = "fr"
DEFAULT_MT_MAX_LENGTH = 256
DEFAULT_SAMPLE_RATE = 16000


class CascadePipelineNotReadyError(RuntimeError):
    """Backend cascade demandé mais non supporté dans ce dépôt."""


@dataclass(frozen=True)
class CascadeSettings:
    """
    Paramètres résolus pour un run cascade (issus du YAML + défauts).

    Attributs :
        asr_backend : Identifiant logique du moteur ASR (ex. ``whisper``).
        asr_model_id : Modèle Hugging Face ou chemin checkpoint ASR.
        mt_backend : Identifiant logique du moteur MT (ex. ``marian``).
        mt_model_id : Modèle Hugging Face ou chemin checkpoint MT.
        asr_language : Langue source attendue pour l'ASR (défaut ``fr``).
        mt_max_length : Longueur max de génération MT (tokens).
    """

    asr_backend: str
    asr_model_id: str
    mt_backend: str
    mt_model_id: str
    asr_language: str
    mt_max_length: int


def resolve_cascade_config_path(path: Path) -> Path:
    """
    Résoudre un chemin de config YAML relatif au dépôt ou à ``4_cascade/configs/``.

    Paramètres :
        path : Chemin fourni par l'utilisateur (relatif ou absolu).

    Retour :
        Chemin absolu vers le fichier YAML existant.

    Lève :
        FileNotFoundError : si le fichier n'existe pas.
    """
    candidate = Path(path)
    if candidate.is_file():
        return candidate.resolve()
    under_cascade = (CASCADE_ROOT / "configs" / candidate).resolve()
    if under_cascade.is_file():
        return under_cascade
    under_project = (PROJECT_ROOT / candidate).resolve()
    if under_project.is_file():
        return under_project
    raise FileNotFoundError(f"Cascade config not found: {path}")


def load_cascade_settings(config: dict[str, Any]) -> CascadeSettings:
    """
    Extraire les champs ASR/MT/décodage depuis un dict YAML chargé.

    Paramètres :
        config : Configuration d'expérience (racine YAML).

    Retour :
        Objet ``CascadeSettings`` prêt pour les appels ASR puis MT.
    """
    return CascadeSettings(
        asr_backend=str(deep_get(config, "asr.backend", DEFAULT_ASR_BACKEND)),
        asr_model_id=str(deep_get(config, "asr.model_id", DEFAULT_ASR_MODEL_ID)),
        mt_backend=str(deep_get(config, "mt.backend", DEFAULT_MT_BACKEND)),
        mt_model_id=str(deep_get(config, "mt.model_id", DEFAULT_MT_MODEL_ID)),
        asr_language=str(deep_get(config, "decode.asr_language", DEFAULT_ASR_LANGUAGE)),
        mt_max_length=int(
            deep_get(config, "decode.mt_max_length", DEFAULT_MT_MAX_LENGTH)
        ),
    )


def _resolve_device() -> torch.device:
    """Choisir CUDA si disponible, sinon CPU (evaluate peut être long sur CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Envoyer un batch HF sur ``device`` en conservant le dtype des tenseurs entiers.

    Les ``input_ids`` / masques doivent rester entiers ; seuls les tenseurs flottants
    peuvent être castés (non utilisé ici : modèles cascade chargés en float32).
    """
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=torch.float32)
        else:
            moved[key] = value.to(device=device)
    return moved


class _WhisperAsrEngine:
    """ASR français via Whisper (transformers)."""

    def __init__(self, model_id: str, language: str, device: torch.device) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        # float32 explicite : évite l'écart input float32 / poids fp16 sur CUDA.
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
        )
        self.model.to(device)
        self.model.eval()
        self.asr_language = language.lower().strip() or "fr"

    def transcribe_file(self, audio_path: Path, sample_rate: int) -> str:
        """
        Transcrire un WAV mono au taux attendu (sortie ``2_prepare``).

        Retour :
            Transcription française (texte brut).
        """
        waveform = load_waveform(audio_path, sample_rate)
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            device=self.device, dtype=torch.float32
        )
        with torch.inference_mode():
            token_ids = self.model.generate(
                input_features,
                language=self.asr_language,
                task="transcribe",
            )
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[
            0
        ].strip()


class _MarianMtEngine:
    """Traduction FR→EN via Marian (transformers)."""

    def __init__(self, model_id: str, device: torch.device) -> None:
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device
        self.tokenizer = MarianTokenizer.from_pretrained(model_id)
        self.model = MarianMTModel.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.to(device)
        self.model.eval()

    def translate(self, text_fr: str, max_length: int) -> str:
        """
        Traduire une chaîne française vers l'anglais.

        Paramètres :
            text_fr : Texte source (sortie ASR).
            max_length : Plafond de tokens générés.

        Retour :
            Hypothèse anglaise.
        """
        inputs = self.tokenizer(
            text_fr,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = _move_batch_to_device(inputs, self.device)
        with torch.inference_mode():
            token_ids = self.model.generate(
                **inputs,
                max_length=max_length,
            )
        return self.tokenizer.decode(token_ids[0], skip_special_tokens=True).strip()


# Cache processus : évite de recharger Whisper/Marian à chaque segment du manifest.
_ASR_ENGINES: dict[tuple[str, str, str], _WhisperAsrEngine] = {}
_MT_ENGINES: dict[tuple[str, str], _MarianMtEngine] = {}


def _get_whisper_engine(settings: CascadeSettings) -> _WhisperAsrEngine:
    """Retourner (ou créer) le moteur Whisper pour ``settings``."""
    device_name = _resolve_device().type
    key = (settings.asr_backend, settings.asr_model_id, device_name)
    engine = _ASR_ENGINES.get(key)
    if engine is None:
        engine = _WhisperAsrEngine(
            settings.asr_model_id,
            settings.asr_language,
            _resolve_device(),
        )
        _ASR_ENGINES[key] = engine
    return engine


def _get_marian_engine(settings: CascadeSettings) -> _MarianMtEngine:
    """Retourner (ou créer) le moteur Marian pour ``settings``."""
    device_name = _resolve_device().type
    key = (settings.mt_model_id, device_name)
    engine = _MT_ENGINES.get(key)
    if engine is None:
        engine = _MarianMtEngine(settings.mt_model_id, _resolve_device())
        _MT_ENGINES[key] = engine
    return engine


def clear_cascade_model_cache() -> None:
    """Vider le cache HF (tests unitaires avec mocks)."""
    _ASR_ENGINES.clear()
    _MT_ENGINES.clear()


def transcribe_french(audio_path: Path, settings: CascadeSettings) -> str:
    """
    Étape ASR : convertir un WAV français en transcription texte.

    Paramètres :
        audio_path : WAV 16 kHz mono (sortie ``2_prepare``).
        settings : Backends et modèles ASR.

    Retour :
        Transcription française (chaîne).

    Lève :
        CascadePipelineNotReadyError : backend ASR non supporté.
    """
    if settings.asr_backend != "whisper":
        raise CascadePipelineNotReadyError(
            f"ASR backend {settings.asr_backend!r} non supporté "
            f"(implémenté : 'whisper')."
        )
    return _get_whisper_engine(settings).transcribe_file(
        audio_path,
        DEFAULT_SAMPLE_RATE,
    )


def translate_french_to_english(text_fr: str, settings: CascadeSettings) -> str:
    """
    Étape MT : traduire une transcription française vers l'anglais.

    Paramètres :
        text_fr : Texte source (sortie ASR ou référence intermédiaire).
        settings : Backends et modèles MT.

    Retour :
        Hypothèse anglaise (chaîne).

    Lève :
        CascadePipelineNotReadyError : backend MT non supporté.
    """
    if settings.mt_backend != "marian":
        raise CascadePipelineNotReadyError(
            f"MT backend {settings.mt_backend!r} non supporté (implémenté : 'marian')."
        )
    stripped = text_fr.strip()
    if not stripped:
        return ""
    return _get_marian_engine(settings).translate(
        stripped,
        settings.mt_max_length,
    )


def cascade_translate_audio(audio_path: Path, settings: CascadeSettings) -> str:
    """
    Enchaîner ASR puis MT sur un fichier audio (contrat inférence / evaluate).

    Paramètres :
        audio_path : Chemin WAV d'entrée.
        settings : Configuration cascade résolue.

    Retour :
        Traduction anglaise finale.

    Lève :
        CascadePipelineNotReadyError : si l'une des deux étapes n'est pas câblée.
    """
    transcript_fr = transcribe_french(audio_path, settings)
    return translate_french_to_english(transcript_fr, settings)
