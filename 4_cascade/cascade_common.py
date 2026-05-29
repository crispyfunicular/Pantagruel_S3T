#!/usr/bin/env python3
"""
Bibliothèque partagée — baseline cascade ASR→MT (fr → en).

Rôle dans le pipeline S3T :
- charger la configuration YAML (`4_cascade/configs/...`) ;
- enchaîner ASR (audio FR → texte FR) puis MT (texte FR → texte EN) ;
- exposer des fonctions réutilisables par `evaluate_cascade.py` et `infer_cascade.py`.

État actuel : **squelette** — le routeur CLI et le dry-run sont en place ; les backends
ASR/MT (Whisper, Marian, etc.) restent à câbler (voir `4_cascade/README.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speechLLM.speechllm_common import PROJECT_ROOT, deep_get

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


class CascadePipelineNotReadyError(RuntimeError):
    """Le pipeline ASR ou MT n'est pas encore branché dans ce dépôt."""


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


def transcribe_french(audio_path: Path, settings: CascadeSettings) -> str:
    """
    Étape ASR : convertir un WAV français en transcription texte.

    Paramètres :
        audio_path : WAV 16 kHz mono (sortie ``2_prepare``).
        settings : Backends et modèles ASR.

    Retour :
        Transcription française (chaîne).

    Lève :
        CascadePipelineNotReadyError : tant que le backend ASR n'est pas implémenté.
    """
    _ = (audio_path, settings)
    raise CascadePipelineNotReadyError(
        f"ASR backend {settings.asr_backend!r} ({settings.asr_model_id}) "
        "n'est pas encore implémenté. Voir 4_cascade/README.md § Backends."
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
        CascadePipelineNotReadyError : tant que le backend MT n'est pas implémenté.
    """
    _ = text_fr
    raise CascadePipelineNotReadyError(
        f"MT backend {settings.mt_backend!r} ({settings.mt_model_id}) "
        "n'est pas encore implémenté. Voir 4_cascade/README.md § Backends."
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
