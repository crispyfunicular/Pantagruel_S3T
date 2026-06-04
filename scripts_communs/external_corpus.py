"""
Corpus audio externes (hors m-TEDx) — manifest TSV pour inférence ST fr→en.

Utilisé pour le corpus pluriTAL / oralité (WAV + .lab français) : pas de références
anglaises → pas de SacreBLEU, seulement inférence + transcription FR en métadonnées.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

# Colonnes TSV alignées sur ``read_manifest`` (``tgt_text`` vide si pas de ref EN).
MANIFEST_COLUMNS = (
    "id",
    "audio",
    "n_frames",
    "tgt_text",
    "speaker",
    "tgt_lang",
    "src_text",
    "src_lang",
)


@dataclass(frozen=True)
class ExternalAudioItem:
    """Un clip WAV externe avec transcription française optionnelle (.lab)."""

    sample_id: str
    audio_path: Path
    src_text: str


def read_french_lab(lab_path: Path) -> str:
    """
    Lire une référence française depuis un fichier ``.lab`` (une phrase par fichier).

    Paramètres :
        lab_path : Chemin vers ``<id>.lab``.

    Retour :
        Texte source normalisé (espaces simples) ou chaîne vide si absent.
    """
    if not lab_path.is_file():
        return ""
    return " ".join(lab_path.read_text(encoding="utf-8").split())


def discover_wav_lab_pairs(corpus_dir: Path) -> list[ExternalAudioItem]:
    """
    Lister les paires ``<id>.wav`` + ``<id>.lab`` dans un répertoire plat.

    Paramètres :
        corpus_dir : Dossier contenant les fichiers (ex. oralité ``corpus/``).

    Retour :
        Liste triée par ``sample_id`` ; seuls les fichiers ``*.wav`` sont retenus.
    """
    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.is_dir():
        msg = f"Corpus directory not found: {corpus_dir}"
        raise FileNotFoundError(msg)

    items: list[ExternalAudioItem] = []
    for wav_path in sorted(corpus_dir.glob("*.wav")):
        sample_id = wav_path.stem
        if not re.match(r"^[0-9]+-[0-9]+$", sample_id):
            # Ex. clips isolés ``nasal.lab`` sans WAV — ignorés ici.
            pass
        lab_path = corpus_dir / f"{sample_id}.lab"
        items.append(
            ExternalAudioItem(
                sample_id=sample_id,
                audio_path=wav_path.resolve(),
                src_text=read_french_lab(lab_path),
            )
        )
    return items


def estimate_n_frames(audio_path: Path, sample_rate: int = 16000) -> int:
    """
    Estimer ``n_frames`` (échantillons mono) pour le manifest, sans charger tout le signal.

    Paramètres :
        audio_path : Fichier WAV.
        sample_rate : Hz attendus (16 kHz pour S3T).

    Retour :
        Nombre d'échantillons mono ou 0 si lecture impossible.
    """
    try:
        import soundfile as sf

        info = sf.info(audio_path.as_posix())
        if int(info.samplerate) != sample_rate:
            return 0
        channels = max(1, int(info.channels))
        return int(info.frames) * channels // channels
    except (OSError, RuntimeError):
        return 0


def write_external_manifest(
    items: list[ExternalAudioItem],
    output_path: Path,
    *,
    sample_rate: int = 16000,
    speaker: str = "oralite",
) -> Path:
    """
    Écrire un manifest TSV compatible ``read_manifest`` (``tgt_text`` vide).

    Paramètres :
        items : Échantillons découverts.
        output_path : Fichier ``.tsv`` de sortie.
        sample_rate : Pour le champ ``n_frames``.
        speaker : Identifiant locuteur factice pour la colonne ``speaker``.

    Retour :
        Chemin du manifest écrit.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "id": item.sample_id,
                    "audio": str(item.audio_path),
                    "n_frames": estimate_n_frames(item.audio_path, sample_rate),
                    "tgt_text": "",
                    "speaker": speaker,
                    "tgt_lang": "en",
                    "src_text": item.src_text,
                    "src_lang": "fr",
                }
            )
    return output_path
