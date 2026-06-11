"""
Corpus personnel — lectures FR avec références anglaises (évaluation hors m-TEDx).

Entrées typiques :
    - ``corpus_perso/*.wav`` (16 kHz mono, nommage ``N-K.wav``)
    - ``corpus_perso/corpus_perso_ref_EN.txt`` (200 phrases EN, 40 paragraphes × 5)

Sortie :
    - Manifest TSV compatible ``read_manifest`` (``tgt_text`` = paragraphe EN de référence).

Convention : ``N-K.wav`` correspond au paragraphe d'indice ``(N-1)*2 + (K-1)`` (textes distincts).
"""

from __future__ import annotations

import csv
import re
import wave
from dataclasses import dataclass
from pathlib import Path

from scripts_communs.external_corpus import MANIFEST_COLUMNS

PARAGRAPH_MARKER_RE = re.compile(r"Paragraphs\s+\d+\s*-\s*\d+", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")
WAV_NAME_RE = re.compile(r"^(\d+)-(\d+)\.wav$")


@dataclass(frozen=True)
class CorpusPersoParagraph:
    """Un paragraphe de référence (5 phrases anglaises jointes)."""

    index: int
    text: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class CorpusPersoItem:
    """Un clip WAV du corpus perso et sa référence anglaise."""

    sample_id: str
    audio_path: Path
    paragraph_index: int
    tgt_text: str
    n_frames: int


def parse_reference_paragraphs(reference_path: Path) -> list[CorpusPersoParagraph]:
    """
    Parser ``corpus_perso_ref_EN.txt`` en 40 paragraphes de 5 phrases.

    Les marqueurs ``Paragraphs X - Y`` sont retirés ; les phrases sont séparées
    par une ponctuation finale suivie de deux espaces ou plus.

    Paramètres :
        reference_path : Fichier texte de référence anglaise.

    Retour :
        Liste de 40 paragraphes (index 0-based).

    Lève :
        ValueError : Nombre de phrases ou de paragraphes inattendu.
    """
    raw = reference_path.read_text(encoding="utf-8").replace("\n", " ")
    blocks = [
        block.strip() for block in PARAGRAPH_MARKER_RE.split(raw) if block.strip()
    ]
    if not blocks:
        msg = f"Aucun bloc de texte dans {reference_path}"
        raise ValueError(msg)

    sentences: list[str] = []
    for block_idx, block in enumerate(blocks):
        parts = [
            part.strip() for part in SENTENCE_SPLIT_RE.split(block) if part.strip()
        ]
        if len(parts) != 25:
            msg = (
                f"Bloc {block_idx}: attendu 25 phrases, trouvé {len(parts)} "
                f"dans {reference_path}"
            )
            raise ValueError(msg)
        sentences.extend(parts)

    if len(sentences) != 200:
        msg = f"Attendu 200 phrases, trouvé {len(sentences)} dans {reference_path}"
        raise ValueError(msg)

    paragraphs: list[CorpusPersoParagraph] = []
    for idx in range(0, len(sentences), 5):
        chunk = tuple(sentences[idx : idx + 5])
        paragraphs.append(
            CorpusPersoParagraph(
                index=len(paragraphs),
                text=" ".join(chunk),
                sentences=chunk,
            )
        )

    if len(paragraphs) != 40:
        msg = f"Attendu 40 paragraphes, trouvé {len(paragraphs)}"
        raise ValueError(msg)
    return paragraphs


def wav_id_to_paragraph_index(sample_id: str) -> int:
    """
    Convertir un identifiant ``N-K`` en index de paragraphe (0-based).

    Paramètres :
        sample_id : Identifiant sans extension (ex. ``3-2``).

    Retour :
        Index du paragraphe correspondant.

    Lève :
        ValueError : Format d'identifiant invalide ou hors bornes.
    """
    match = WAV_NAME_RE.match(f"{sample_id}.wav")
    if not match:
        msg = f"Identifiant WAV invalide: {sample_id!r} (attendu N-K)"
        raise ValueError(msg)
    n = int(match.group(1))
    k = int(match.group(2))
    if not (1 <= n <= 20 and k in {1, 2}):
        msg = f"Identifiant hors bornes: {sample_id!r}"
        raise ValueError(msg)
    return (n - 1) * 2 + (k - 1)


def estimate_n_frames_wav(audio_path: Path) -> int:
    """Lire ``n_frames`` depuis un WAV mono 16 kHz (stdlib ``wave``)."""
    try:
        with wave.open(audio_path.as_posix(), "rb") as handle:
            if handle.getnchannels() != 1:
                return 0
            if handle.getframerate() != 16000:
                return 0
            return int(handle.getnframes())
    except (OSError, wave.Error):
        return 0


def discover_corpus_perso_wavs(corpus_dir: Path) -> list[Path]:
    """
    Lister les WAV ``N-K.wav`` du dossier corpus perso.

    Paramètres :
        corpus_dir : Répertoire contenant les fichiers audio.

    Retour :
        Chemins triés par (N, K).
    """
    corpus_dir = corpus_dir.resolve()
    if not corpus_dir.is_dir():
        msg = f"Dossier corpus introuvable: {corpus_dir}"
        raise FileNotFoundError(msg)

    wavs = [path for path in corpus_dir.glob("*.wav") if WAV_NAME_RE.match(path.name)]
    return sorted(
        wavs,
        key=lambda path: (
            int(WAV_NAME_RE.match(path.name).group(1)),  # type: ignore[union-attr]
            int(WAV_NAME_RE.match(path.name).group(2)),  # type: ignore[union-attr]
        ),
    )


def build_corpus_perso_items(
    *,
    corpus_dir: Path,
    reference_path: Path,
    project_root: Path,
) -> list[CorpusPersoItem]:
    """
    Associer chaque WAV à son paragraphe de référence anglais.

    Paramètres :
        corpus_dir : Dossier ``corpus_perso/`` (WAVs).
        reference_path : Fichier ``corpus_perso_ref_EN.txt``.
        project_root : Racine du dépôt (chemins audio relatifs).

    Retour :
        Liste triée des échantillons prêts pour le manifest.
    """
    paragraphs = parse_reference_paragraphs(reference_path)
    wav_paths = discover_corpus_perso_wavs(corpus_dir)
    if len(wav_paths) != 40:
        msg = f"Attendu 40 WAV, trouvé {len(wav_paths)} dans {corpus_dir}"
        raise ValueError(msg)

    items: list[CorpusPersoItem] = []
    for wav_path in wav_paths:
        sample_id = wav_path.stem
        para_idx = wav_id_to_paragraph_index(sample_id)
        if para_idx >= len(paragraphs):
            msg = f"Paragraphe manquant pour {sample_id} (index {para_idx})"
            raise ValueError(msg)
        try:
            rel_audio = wav_path.resolve().relative_to(project_root.resolve())
            audio_ref = rel_audio.as_posix()
        except ValueError:
            audio_ref = wav_path.resolve().as_posix()

        items.append(
            CorpusPersoItem(
                sample_id=sample_id,
                audio_path=Path(audio_ref),
                paragraph_index=para_idx,
                tgt_text=paragraphs[para_idx].text,
                n_frames=estimate_n_frames_wav(wav_path),
            )
        )
    return items


def write_corpus_perso_manifest(
    items: list[CorpusPersoItem],
    output_path: Path,
    *,
    speaker: str = "corpus_perso",
) -> Path:
    """
    Écrire le manifest TSV d'évaluation (références EN présentes).

    Paramètres :
        items : Échantillons construits par ``build_corpus_perso_items``.
        output_path : Fichier ``.tsv`` de sortie.
        speaker : Identifiant locuteur pour la colonne ``speaker``.

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
                    "n_frames": item.n_frames,
                    "tgt_text": item.tgt_text,
                    "speaker": speaker,
                    "tgt_lang": "en",
                    "src_text": "",
                    "src_lang": "fr",
                }
            )
    return output_path


__all__ = [
    "CorpusPersoItem",
    "CorpusPersoParagraph",
    "build_corpus_perso_items",
    "discover_corpus_perso_wavs",
    "parse_reference_paragraphs",
    "wav_id_to_paragraph_index",
    "write_corpus_perso_manifest",
]
