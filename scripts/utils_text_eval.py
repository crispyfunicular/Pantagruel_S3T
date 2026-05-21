"""
Normalisation de texte et métriques d'erreur ASR (WER/CER) pour tests locaux.

Utilisé par ``quick_eval_hf_asr.py`` pour scorer les hypothèses Whisper par rapport
aux références ``.lab``. Non utilisé pour l'évaluation ST SacreBLEU (l'étape 5 utilise
sacrebleu directement).

Entrées : chaînes de référence/hypothèse brutes.
Sorties : chaînes normalisées et structures ``ErrorRate`` avec comptages d'éditions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Retirer la ponctuation pour une comparaison mot/caractère équitable (compatible Unicode).
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text_for_eval(
    text: str,
    *,
    mode: str = "nfkc",
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """
    Normaliser les chaînes de référence et d'hypothèse avant WER/CER.

    S'aligne sur les normes texte du PRD le cas échéant (NFKC, minuscules optionnelles).

    Paramètres :
        text : Transcription brute.
        mode : ``"nfkc"`` applique la normalisation Unicode NFKC ; ``"none"`` ignore.
        lowercase : Mettre en minuscules pour le score.
        remove_punctuation : Remplacer la ponctuation par des espaces et fusionner les blancs.

    Retour :
        Chaîne normalisée sur une seule ligne.
    """
    value = text.strip()
    if mode == "nfkc":
        value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.split())
    if lowercase:
        value = value.lower()
    if remove_punctuation:
        value = _PUNCT_RE.sub(" ", value)
        value = " ".join(value.split())
    return value


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """
    Distance de Levenshtein au niveau token ou caractère (insertion/suppression/substitution = 1).

    Paramètres :
        ref : Séquence de référence (tokens ou caractères).
        hyp : Séquence hypothèse.

    Retour :
        Distance d'édition minimale.
    """
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, ref_tok in enumerate(ref, start=1):
        curr = [i]
        for j, hyp_tok in enumerate(hyp, start=1):
            cost = 0 if ref_tok == hyp_tok else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


@dataclass(frozen=True)
class ErrorRate:
    """Taux d'erreur mot ou caractère avec comptages bruts pour le rapport."""

    errors: int
    ref_length: int
    rate: float


def word_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """
    Calculer le taux d'erreur de mots (WER) après tokenisation par espaces.

    Paramètres :
        reference : Transcription de référence.
        hypothesis : Transcription système.

    Retour :
        ``ErrorRate`` avec ``rate = errors / len(ref_tokens)`` (1,0 si ref vide et errors>0).
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    errors = _edit_distance(ref_tokens, hyp_tokens)
    ref_len = len(ref_tokens)
    rate = errors / ref_len if ref_len else (0.0 if not errors else 1.0)
    return ErrorRate(errors=errors, ref_length=ref_len, rate=rate)


def char_error_rate(reference: str, hypothesis: str) -> ErrorRate:
    """
    Calculer le taux d'erreur de caractères (CER) sur des chaînes sans espaces.

    Paramètres :
        reference : Transcription de référence.
        hypothesis : Transcription système.

    Retour :
        ``ErrorRate`` sur des listes de caractères.
    """
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    errors = _edit_distance(ref_chars, hyp_chars)
    ref_len = len(ref_chars)
    rate = errors / ref_len if ref_len else (0.0 if not errors else 1.0)
    return ErrorRate(errors=errors, ref_length=ref_len, rate=rate)
