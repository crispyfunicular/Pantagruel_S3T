#!/usr/bin/env python3
"""
Client Gemini — baseline ST (Speech-to-Text Translation) via API.

Ce module fournit une couche d'accès minimale et testable à Gemini pour :
- authentification (clé API via variable d'environnement) ;
- construction d'une requête audio + prompt ;
- appel modèle (Flash/Pro) et extraction du texte.

La baseline est conçue pour produire des sorties comparables à `2_speechLLM/evaluate.py` :
des hypothèses (anglais) évaluées par SacreBLEU contre `tgt_text` des manifests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GEMINI_MODEL_ID = "gemini-2.5-flash"
DEFAULT_PROMPT = "Translate the French speech to English."
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"


class MissingGeminiApiKeyError(RuntimeError):
    """Erreur levée si la clé API Gemini est absente de l'environnement."""


@dataclass(frozen=True)
class GeminiRequest:
    """Paramètres minimaux d'une requête ST Gemini (audio → texte EN)."""

    model_id: str
    prompt: str
    temperature: float = 0.0
    max_output_tokens: int = 256


def get_gemini_api_key() -> str:
    """
    Lire la clé API Gemini depuis l'environnement.

    Retour :
        La clé sous forme de chaîne non vide.

    Lève :
        MissingGeminiApiKeyError : si la variable d'environnement est absente.
    """

    key = (os.getenv(ENV_GEMINI_API_KEY) or "").strip()
    if not key:
        raise MissingGeminiApiKeyError(
            f"Missing {ENV_GEMINI_API_KEY}. Please export it, e.g. "
            f"`export {ENV_GEMINI_API_KEY}=...`."
        )
    return key


def create_gemini_client(*, api_key: str | None = None) -> Any:
    """
    Créer un client Gemini (google-genai).

    Paramètres :
        api_key : Clé API explicite (sinon lecture env).

    Retour :
        Instance client du SDK google-genai.
    """

    if api_key is None:
        api_key = get_gemini_api_key()

    # Import local pour éviter de rendre l'import global cassant si le SDK n'est pas installé.
    from google import genai  # type: ignore[import-not-found]

    return genai.Client(api_key=api_key)


def _guess_audio_mime_type(path: Path) -> str:
    """
    Deviner le mime-type audio.

    Note: la pipeline `prepare` produit des WAV 16 kHz mono ; on se limite donc à wav par défaut.
    """

    suffix = path.suffix.lower().lstrip(".")
    if suffix == "wav":
        return "audio/wav"
    if suffix == "mp3":
        return "audio/mpeg"
    if suffix == "flac":
        return "audio/flac"
    return "application/octet-stream"


def translate_audio(
    *,
    client: Any,
    request: GeminiRequest,
    audio_path: Path,
) -> str:
    """
    Traduire un audio FR en texte EN via Gemini.

    Paramètres :
        client : Client google-genai.
        request : Paramètres modèle/prompt/génération.
        audio_path : Chemin vers le fichier audio.

    Retour :
        Traduction anglaise (chaîne).
    """

    audio_path = Path(audio_path)
    audio_bytes = audio_path.read_bytes()
    mime_type = _guess_audio_mime_type(audio_path)

    # Import local pour isoler la dépendance.
    from google.genai import types  # type: ignore[import-not-found]

    contents = [
        request.prompt,
        types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
    ]
    config = types.GenerateContentConfig(
        temperature=float(request.temperature),
        max_output_tokens=int(request.max_output_tokens),
    )
    response = client.models.generate_content(
        model=request.model_id,
        contents=contents,
        config=config,
    )

    # SDK: le texte est exposé via `response.text` dans la majorité des cas.
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()

    # Fallback robuste si `text` est absent.
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        parts: list[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            cand_parts = (
                getattr(content, "parts", None) if content is not None else None
            )
            if isinstance(cand_parts, list):
                for part in cand_parts:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        parts.append(part_text.strip())
        if parts:
            return "\n".join(parts).strip()

    return ""
