#!/usr/bin/env python3
"""
Bibliothèque partagée — baselines ST open source (variante 6).

Rôle dans le pipeline S3T :
- charger la configuration YAML (`6_open_baselines/configs/...`) ;
- traduire audio français → texte anglais via un modèle pré-entraîné open weights ;
- exposer des fonctions réutilisables par `evaluate_open.py` et `infer_open.py`.

Backends implémentés :
- ``whisper_st`` : Whisper ``task=translate`` (HF transformers).
- ``seamless_m4t_v2`` : Meta SeamlessM4T v2 (HF transformers).
- ``canary_1b`` : NVIDIA Canary-1B (NeMo ``EncDecMultiTaskModel``, dépendance optionnelle).

Les modèles sont chargés une fois par couple (type, model_id) et réutilisés sur tout un run evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from scripts_communs.st_common import load_waveform
from speechLLM.speechllm_lib import PROJECT_ROOT, deep_get

OPEN_BASELINES_ROOT = Path(__file__).resolve().parent

# Codes de sortie documentés pour les scripts open baselines.
EXIT_SUCCESS = 0
EXIT_CONFIG = 2
EXIT_NOT_IMPLEMENTED = 3

DEFAULT_MODEL_TYPE = "whisper_st"
DEFAULT_MODEL_ID_WHISPER = "openai/whisper-large-v3"
DEFAULT_MODEL_ID_SEAMLESS = "facebook/seamless-m4t-v2-large"
DEFAULT_MODEL_ID_CANARY = "nvidia/canary-1b-v2"
DEFAULT_SRC_LANG = "fra"
DEFAULT_TGT_LANG = "eng"
DEFAULT_WHISPER_LANGUAGE = "fr"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MAX_NEW_TOKENS = 256
SUPPORTED_MODEL_TYPES = frozenset({"whisper_st", "seamless_m4t_v2", "canary_1b"})


class OpenPipelineNotReadyError(RuntimeError):
    """Backend open baseline demandé mais non disponible dans cet environnement."""


@dataclass(frozen=True)
class OpenModelSettings:
    """
    Paramètres résolus pour un run open baseline (issus du YAML + défauts).

    Attributs :
        model_type : Identifiant logique du backend (``whisper_st``, etc.).
        model_id : Modèle Hugging Face / NeMo.
        src_lang : Langue source (ISO 639-3 pour Seamless/Canary, ignoré par Whisper-ST).
        tgt_lang : Langue cible (ISO 639-3 pour Seamless, ``en`` pour Canary).
        whisper_language : Code ISO 639-1 pour Whisper (défaut ``fr``).
        device : ``cuda`` ou ``cpu`` (résolu à l'exécution si ``auto``).
        dtype : ``float16``, ``float32`` ou ``bfloat16``.
        max_new_tokens : Plafond de génération (Whisper / Seamless).
    """

    model_type: str
    model_id: str
    src_lang: str
    tgt_lang: str
    whisper_language: str
    device: str
    dtype: str
    max_new_tokens: int


def resolve_open_config_path(path: Path) -> Path:
    """
    Résoudre un chemin de config YAML relatif au dépôt ou à ``6_open_baselines/configs/``.

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
    under_open = (OPEN_BASELINES_ROOT / "configs" / candidate).resolve()
    if under_open.is_file():
        return under_open
    under_project = (PROJECT_ROOT / candidate).resolve()
    if under_project.is_file():
        return under_project
    raise FileNotFoundError(f"Open baselines config not found: {path}")


def _default_model_id(model_type: str) -> str:
    """Retourner l'identifiant HF/NeMo par défaut pour un ``model_type``."""
    if model_type == "seamless_m4t_v2":
        return DEFAULT_MODEL_ID_SEAMLESS
    if model_type == "canary_1b":
        return DEFAULT_MODEL_ID_CANARY
    return DEFAULT_MODEL_ID_WHISPER


def load_open_settings(config: dict[str, Any]) -> OpenModelSettings:
    """
    Extraire les champs modèle/décodage depuis un dict YAML chargé.

    Paramètres :
        config : Configuration d'expérience (racine YAML).

    Retour :
        Objet ``OpenModelSettings`` prêt pour l'inférence ST open.
    """
    model_type = str(deep_get(config, "model.type", DEFAULT_MODEL_TYPE)).strip()
    return OpenModelSettings(
        model_type=model_type,
        model_id=str(deep_get(config, "model.model_id", _default_model_id(model_type))),
        src_lang=str(deep_get(config, "model.src_lang", DEFAULT_SRC_LANG)),
        tgt_lang=str(deep_get(config, "model.tgt_lang", DEFAULT_TGT_LANG)),
        whisper_language=str(
            deep_get(config, "model.whisper_language", DEFAULT_WHISPER_LANGUAGE)
        ),
        device=str(deep_get(config, "model.device", "auto")),
        dtype=str(deep_get(config, "model.dtype", "float16")),
        max_new_tokens=int(
            deep_get(config, "decode.max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        ),
    )


def _resolve_device(requested: str) -> torch.device:
    """Choisir CUDA si disponible (sauf si ``cpu`` forcé)."""
    normalized = requested.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized in {"cuda", "auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """Convertir une chaîne YAML en ``torch.dtype``."""
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.strip().lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name!r}")
    return mapping[key]


def _move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Envoyer un batch HF sur ``device`` en castant les tenseurs flottants."""
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


class _WhisperStEngine:
    """ST directe fr→en via Whisper ``task=translate`` (transformers)."""

    def __init__(
        self,
        model_id: str,
        language: str,
        device: torch.device,
        dtype: torch.dtype,
        max_new_tokens: int,
    ) -> None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.processor = WhisperProcessor.from_pretrained(model_id)
        # Whisper : float32 stable sur V100 ; fp16 optionnel via YAML.
        load_dtype = dtype if device.type == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
        )
        self.model.to(device)
        self.model.eval()
        self.whisper_language = language.lower().strip() or "fr"

    def translate_file(self, audio_path: Path, sample_rate: int) -> str:
        """Traduire un WAV français vers l'anglais (sortie texte brute)."""
        waveform = load_waveform(audio_path, sample_rate)
        inputs = self.processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=self.device)
        if input_features.is_floating_point():
            input_features = input_features.to(dtype=self.model.dtype)
        with torch.inference_mode():
            token_ids = self.model.generate(
                input_features,
                language=self.whisper_language,
                task="translate",
                max_new_tokens=self.max_new_tokens,
            )
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[
            0
        ].strip()


class _SeamlessM4Tv2Engine:
    """ST directe multilingue via SeamlessM4T v2 (transformers)."""

    def __init__(
        self,
        model_id: str,
        src_lang: str,
        tgt_lang: str,
        device: torch.device,
        dtype: torch.dtype,
        max_new_tokens: int,
    ) -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2Model

        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_new_tokens = max_new_tokens
        load_dtype = dtype if device.type == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SeamlessM4Tv2Model.from_pretrained(
            model_id,
            torch_dtype=load_dtype,
        )
        self.model.to(device)
        self.model.eval()

    def translate_file(self, audio_path: Path, sample_rate: int) -> str:
        """Traduire un WAV via SeamlessM4T v2 (génération texte uniquement)."""
        waveform = load_waveform(audio_path, sample_rate)
        inputs = self.processor(
            audios=waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        inputs = _move_batch_to_device(inputs, self.device, self.model.dtype)
        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                tgt_lang=self.tgt_lang,
                generate_speech=False,
                max_new_tokens=self.max_new_tokens,
            )
        return self.processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True,
        ).strip()


class _Canary1bEngine:
    """ST multilingue via NVIDIA Canary-1B (NeMo, dépendance optionnelle)."""

    def __init__(self, model_id: str) -> None:
        try:
            from nemo.collections.asr.models import EncDecMultiTaskModel
        except ImportError as exc:
            raise OpenPipelineNotReadyError(
                "Backend canary_1b requiert nemo_toolkit[asr] "
                "(pip install 'nemo_toolkit[asr]>=2.0')."
            ) from exc

        self.model = EncDecMultiTaskModel.from_pretrained(model_id)
        self.model.eval()

    def translate_file(
        self,
        audio_path: Path,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        """Traduire un WAV via Canary (``transcribe`` avec langues source/cible)."""
        # Canary attend des codes courts (fr/en) plutôt qu'ISO 639-3.
        source = _canary_lang_code(src_lang)
        target = _canary_lang_code(tgt_lang)
        outputs = self.model.transcribe(
            paths2audio_files=[str(audio_path.resolve())],
            batch_size=1,
            source_lang=source,
            target_lang=target,
        )
        if not outputs:
            return ""
        first = outputs[0]
        text = getattr(first, "text", None)
        if text is None and isinstance(first, str):
            text = first
        return str(text or "").strip()


def _canary_lang_code(lang: str) -> str:
    """Réduire ISO 639-3 (fra/eng) vers le code court attendu par Canary."""
    normalized = lang.strip().lower()
    mapping = {
        "fra": "fr",
        "fre": "fr",
        "fr": "fr",
        "eng": "en",
        "en": "en",
    }
    return mapping.get(normalized, normalized[:2] if normalized else "fr")


# Cache processus : évite de recharger les modèles à chaque segment du manifest.
_WHISPER_ST_ENGINES: dict[tuple[str, str, str, str], _WhisperStEngine] = {}
_SEAMLESS_ENGINES: dict[tuple[str, str, str, str], _SeamlessM4Tv2Engine] = {}
_CANARY_ENGINES: dict[str, _Canary1bEngine] = {}


def clear_open_model_cache() -> None:
    """Vider le cache modèles (tests unitaires avec mocks)."""
    _WHISPER_ST_ENGINES.clear()
    _SEAMLESS_ENGINES.clear()
    _CANARY_ENGINES.clear()


def _get_whisper_st_engine(settings: OpenModelSettings) -> _WhisperStEngine:
    """Retourner (ou créer) le moteur Whisper-ST pour ``settings``."""
    device = _resolve_device(settings.device)
    dtype = _resolve_dtype(settings.dtype)
    key = (
        settings.model_type,
        settings.model_id,
        device.type,
        settings.whisper_language,
    )
    engine = _WHISPER_ST_ENGINES.get(key)
    if engine is None:
        engine = _WhisperStEngine(
            settings.model_id,
            settings.whisper_language,
            device,
            dtype,
            settings.max_new_tokens,
        )
        _WHISPER_ST_ENGINES[key] = engine
    return engine


def _get_seamless_engine(settings: OpenModelSettings) -> _SeamlessM4Tv2Engine:
    """Retourner (ou créer) le moteur SeamlessM4T v2 pour ``settings``."""
    device = _resolve_device(settings.device)
    dtype = _resolve_dtype(settings.dtype)
    key = (
        settings.model_type,
        settings.model_id,
        device.type,
        f"{settings.src_lang}->{settings.tgt_lang}",
    )
    engine = _SEAMLESS_ENGINES.get(key)
    if engine is None:
        engine = _SeamlessM4Tv2Engine(
            settings.model_id,
            settings.src_lang,
            settings.tgt_lang,
            device,
            dtype,
            settings.max_new_tokens,
        )
        _SEAMLESS_ENGINES[key] = engine
    return engine


def _get_canary_engine(settings: OpenModelSettings) -> _Canary1bEngine:
    """Retourner (ou créer) le moteur Canary pour ``settings``."""
    engine = _CANARY_ENGINES.get(settings.model_id)
    if engine is None:
        engine = _Canary1bEngine(settings.model_id)
        _CANARY_ENGINES[settings.model_id] = engine
    return engine


def open_translate_audio(audio_path: Path, settings: OpenModelSettings) -> str:
    """
    Traduire un fichier audio français vers l'anglais (contrat inférence / evaluate).

    Paramètres :
        audio_path : Chemin WAV d'entrée (16 kHz mono recommandé).
        settings : Configuration open baseline résolue.

    Retour :
        Hypothèse anglaise (chaîne).

    Lève :
        OpenPipelineNotReadyError : type de modèle non supporté ou dépendance manquante.
    """
    model_type = settings.model_type.strip().lower()
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise OpenPipelineNotReadyError(
            f"model.type {settings.model_type!r} non supporté "
            f"(implémenté : {sorted(SUPPORTED_MODEL_TYPES)})."
        )
    if model_type == "whisper_st":
        return _get_whisper_st_engine(settings).translate_file(
            audio_path,
            DEFAULT_SAMPLE_RATE,
        )
    if model_type == "seamless_m4t_v2":
        return _get_seamless_engine(settings).translate_file(
            audio_path,
            DEFAULT_SAMPLE_RATE,
        )
    if model_type == "canary_1b":
        return _get_canary_engine(settings).translate_file(
            audio_path,
            settings.src_lang,
            settings.tgt_lang,
        )
    raise OpenPipelineNotReadyError(f"Backend {model_type!r} non câblé.")
