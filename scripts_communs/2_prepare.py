#!/usr/bin/env python3
"""
Étape pipeline S3T 2 — Préparer le corpus m-TEDx pour l'entraînement ST aval.

Rôle dans le pipeline
-------------
Découper les FLAC TEDx longs en clips au niveau utterance, normaliser le texte
parallèle et émettre des manifests pour les étapes 3–6 (tokenizer, SPM, train, evaluate).

Entrées
------
- Arborescence m-TEDx brute sous ``--input-root`` (défaut ``datasets/raw``), soit
  ``<langpair>/data/<split>/`` soit l'ancien ``mtedx_<langpair>/data/<split>/``.
- Par split : ``txt/<split>.yaml`` (offsets/durées), ``txt/<split>.{src,tgt}``,
  et ``wav/*.flac`` référencés par les clés ``wav`` du YAML.

Sorties
-------
- Audio : ``<output-root>/<langpair>/<split>/<utt_id>.wav`` — 16 kHz mono PCM16.
- Manifests : ``<manifests-root>/<langpair>/{train,valid,test}.tsv``.
- Lignes cible : ``<manifests-root>/<langpair>/<split>.target.txt`` (tgt uniquement).
- Rapport JSON : ``artifacts/prepare_<langpair>.json`` ; ``*.progress.json`` optionnel.

Politique anti-fuite
----------------
Après traitement de tous les splits, ``detect_leaks`` signale les chevauchements
d'ids utterance ou de texte cible normalisé entre train et valid/test. Avec ``--fail-on-leak``
par défaut, tout chevauchement fixe le code de sortie 5 ; utiliser ``--no-fail-on-leak`` pour
ne garder que des avertissements.

Codes de sortie
----------
0 — succès (y compris dry-run si la disposition du corpus est valide).
2 — ``--langpair`` invalide ou corpus d'entrée manquant (``run_from_namespace``).
4 — erreurs de traitement par segment (YAML, audio manquant, échecs d'extraction).
5 — fuite train vs valid/test avec ``--fail-on-leak`` (activé par défaut).
6 — vérification WAV a posteriori échouée (``--verify-only`` ou fin de ``run_prepare``).

Usage :
    python scripts_communs/2_prepare.py --langpair fr-en
    python scripts_communs/2_prepare.py --langpair fr-es --dry-run
    python scripts_communs/pipeline.py prepare --langpair fr-en
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_LANGPAIRS = frozenset({"fr-en", "fr-pt", "fr-es"})
SPLITS = ("train", "valid", "test")

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"

SEGMENT_MODES = ("utterance", "sentence_like")

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


@dataclass
class SegmentRecord:
    """Un utterance m-TEDx avant filtrage et extraction WAV."""

    utt_id: str
    talk_id: str
    order_idx: int
    wav_path: Path
    offset_s: float
    duration_s: float
    src_text: str
    tgt_text: str
    speaker: str
    src_lang: str
    tgt_lang: str


@dataclass
class SplitStats:
    """Compteurs par split écrits dans le rapport JSON prepare."""

    split: str
    segments_in: int = 0
    segments_kept: int = 0
    segments_dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)


PROGRESS_INTERVAL = 200


@dataclass
class PrepareReport:
    """Résumé sérialisable d'un run prepare (également persisté en JSON)."""

    timestamp_utc: str
    langpair: str
    input_root: str
    output_root: str
    manifests_root: str
    sample_rate: int
    min_duration: float
    max_duration: float
    text_norm: str
    lowercase: bool
    segment_mode: str = "utterance"
    sentence_like: dict[str, Any] = field(default_factory=dict)
    splits: list[SplitStats] = field(default_factory=list)
    leak_issues: list[str] = field(default_factory=list)
    exit_code: int = 0


def parse_langpair(value: str) -> str:
    """Valider et normaliser une valeur CLI de paire de langues.

    Paramètres :
        value: Raw ``--langpair`` string (e.g. ``fr-en``).

    Retour :
        Paire nettoyée si elle est dans ``SUPPORTED_LANGPAIRS``.

    Lève :
        ValueError : Si la paire n'est pas supportée.
    """
    pair = value.strip()
    if pair not in SUPPORTED_LANGPAIRS:
        supported = ", ".join(sorted(SUPPORTED_LANGPAIRS))
        raise ValueError(f"Unknown langpair: {pair}. Supported: {supported}")
    return pair


def resolve_corpus_root(input_root: Path, langpair: str) -> Path:
    """Localiser l'arborescence m-TEDx extraite sous la racine des données brutes.

    Les archives OpenSLR se décompressent en ``mtedx_<langpair>`` or plain ``<langpair>``;
    le premier candidat dont le répertoire ``data/train`` existe est retenu.

    Paramètres :
        input_root: Root containing downloaded corpora (``datasets/raw``).
        langpair: Language pair slug (e.g. ``fr-en``).

    Retour :
        Path to the corpus root (parent of ``data/``).

    Lève :
        FileNotFoundError : Si aucune disposition ne contient ``data/train``.
    """
    candidates = (
        input_root / f"mtedx_{langpair}",
        input_root / langpair,
    )
    for path in candidates:
        if (path / "data" / "train").is_dir():
            return path
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No m-TEDx corpus for {langpair} under {input_root} (tried: {tried}). "
        "Run stage download first."
    )


def manifest_audio_path(
    output_root: Path,
    langpair: str,
    split: str,
    utt_id: str,
    *,
    repo_root: Path | None = None,
) -> str:
    """Construire le chemin colonne ``audio`` stocké dans les manifests TSV.

    Préfère un chemin POSIX relatif à la racine du dépôt S3T pour des manifests portables ;
    repli sur un chemin absolu si les sorties sont hors de l'arbre projet.

    Paramètres :
        output_root: Processed audio root (``datasets/processed``).
        langpair: Language pair slug.
        split: One of ``train``, ``valid``, ``test``.
        utt_id: Utterance identifier (stem + segment index).
        repo_root: Base for relative paths (defaults to ``PROJECT_ROOT``).

    Retour :
        Forward-slash path string for the manifest ``audio`` field.
    """
    abs_audio = (output_root / langpair / split / f"{utt_id}.wav").resolve()
    base = (repo_root or PROJECT_ROOT).resolve()
    try:
        return abs_audio.relative_to(base).as_posix()
    except ValueError:
        return abs_audio.as_posix()


def normalize_text(text: str, *, mode: str, lowercase: bool) -> str:
    """Appliquer la normalisation texte alignée PRD pour les lignes parallèles src/tgt.

    Paramètres :
        text: Raw transcript line from m-TEDx.
        mode: ``nfkc`` (Unicode NFKC + collapse whitespace) or ``none``.
        lowercase: When True, fold case after normalization.

    Retour :
        Normalized string (may be empty if input was whitespace-only).
    """
    value = text.strip()
    if mode == "nfkc":
        value = unicodedata.normalize("NFKC", value)
    value = " ".join(value.split())
    if lowercase:
        value = value.lower()
    return value


def iter_mtedx_segments(
    root: Path, langpair: str, split: str
) -> Iterator[SegmentRecord]:
    """Produire les métadonnées de segment depuis YAML m-TEDx et fichiers texte parallèles.

    Aligne les indices de liste de segments YAML avec ``.{src_lang}`` / ``.{tgt_lang}`` line
    numbers, puis regroupe les segments par FLAC parent et trie par offset dans chaque fichier.

    Paramètres :
        root: Resolved corpus root (contains ``data/<split>/``).
        langpair: Pair slug used to derive ``src_lang`` and ``tgt_lang``.
        split: ``train``, ``valid``, or ``test``.

    Produit :
        ``SegmentRecord`` instances ready for duration/text filtering.

    Lève :
        FileNotFoundError : ``<split>.yaml`` manquant.
        ValueError : YAML pas une liste, ou décomptes de lignes incohérents entre fichiers.
    """
    src_lang, tgt_lang = langpair.split("-", maxsplit=1)
    data_dir = root / "data" / split
    wav_root = data_dir / "wav"
    txt_root = data_dir / "txt"
    yaml_path = txt_root / f"{split}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing split descriptor: {yaml_path}")

    with yaml_path.open(encoding="utf-8") as handle:
        segments = yaml.safe_load(handle)
    if not isinstance(segments, list):
        raise ValueError(f"Unexpected YAML content in {yaml_path}")

    src_lines = (
        (txt_root / f"{split}.{src_lang}").read_text(encoding="utf-8").splitlines()
    )
    tgt_lines = (
        (txt_root / f"{split}.{tgt_lang}").read_text(encoding="utf-8").splitlines()
    )
    if len(segments) != len(src_lines) or len(segments) != len(tgt_lines):
        raise ValueError(
            f"Line count mismatch in {split}: yaml={len(segments)}, "
            f"{src_lang}={len(src_lines)}, {tgt_lang}={len(tgt_lines)}"
        )

    # Regrouper par FLAC parent : le YAML liste plusieurs segments par long enregistrement.
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, segment in enumerate(segments):
        wav_key = segment["wav"]
        grouped[wav_key].append((index, segment))

    for wav_key in sorted(grouped):
        # Ordre stable des segments dans un talk (requis pour suffixe utt_id _0, _1, …).
        entries = sorted(grouped[wav_key], key=lambda item: float(item[1]["offset"]))
        # OpenSLR fournit du FLAC ; les clés YAML utilisent encore un suffixe .wav.
        flac_name = wav_key.replace(".wav", ".flac")
        wav_path = wav_root / flac_name
        stem = Path(wav_key).stem
        for seg_index, (line_index, segment) in enumerate(entries):
            # line_index lie ce segment à la ligne texte parallèle src/tgt.
            yield SegmentRecord(
                utt_id=f"{stem}_{seg_index}",
                talk_id=stem,
                order_idx=int(seg_index),
                wav_path=wav_path,
                offset_s=float(segment["offset"]),
                duration_s=float(segment["duration"]),
                src_text=src_lines[line_index].strip(),
                tgt_text=tgt_lines[line_index].strip(),
                speaker=str(segment.get("speaker_id", "")),
                src_lang=src_lang,
                tgt_lang=tgt_lang,
            )


def _ends_with_sentence_punctuation(text: str) -> bool:
    """Heuristique simple : détecter une fin de phrase (ponctuation forte)."""

    value = text.strip()
    return bool(value) and value[-1] in ".?!"


def merge_segments_sentence_like(
    segments: list[SegmentRecord],
    *,
    target_duration_s: float,
    max_duration_s: float,
    require_punctuation: bool,
) -> tuple[list[SegmentRecord], dict[str, Any]]:
    """
    Fusionner des segments contigus en unités \"phrase-like\".

    Idée : regrouper des segments du même talk (et, autant que possible, même speaker)
    jusqu'à atteindre une ponctuation forte, tout en bornant la durée.

    Paramètres :
        segments : Liste de segments d'un split (typiquement déjà triée par talk/offset).
        target_duration_s : Durée cible de fusion (couper dès qu'on a une phrase complète
            et qu'on atteint environ cette durée).
        max_duration_s : Durée maximale d'un segment fusionné.
        require_punctuation : Si True, on préfère couper uniquement quand la fusion
            finit sur une ponctuation forte (src ou tgt), sauf si on dépasse max_duration_s.

    Retour :
        (merged_segments, stats) où stats contient des compteurs utiles au rapport.
    """

    if not segments:
        return [], {"segments_in": 0, "segments_out": 0, "merged_groups": 0}

    # Sécurités.
    target_duration_s = float(target_duration_s)
    max_duration_s = float(max_duration_s)
    if target_duration_s <= 0:
        target_duration_s = 10.0
    if max_duration_s <= 0:
        max_duration_s = 15.0
    if target_duration_s > max_duration_s:
        target_duration_s = max_duration_s

    # Grouper par talk_id pour éviter toute fusion inter-talk.
    by_talk: dict[str, list[SegmentRecord]] = defaultdict(list)
    for seg in segments:
        by_talk[seg.talk_id].append(seg)

    merged: list[SegmentRecord] = []
    merged_groups = 0

    for talk_id in sorted(by_talk):
        current_talk_id = talk_id
        talk_segments = sorted(
            by_talk[current_talk_id], key=lambda s: (s.order_idx, s.offset_s)
        )
        group: list[SegmentRecord] = []
        group_duration = 0.0
        group_index = 0
        group_speaker: str | None = None

        def flush_group(*, talk_id: str = current_talk_id) -> None:
            nonlocal group, group_duration, group_index, merged_groups, group_speaker
            if not group:
                return
            first = group[0]
            last = group[-1]
            utt_id = f"{talk_id}_m{group_index}"
            group_index += 1
            merged_groups += 1
            merged.append(
                SegmentRecord(
                    utt_id=utt_id,
                    talk_id=talk_id,
                    order_idx=first.order_idx,
                    wav_path=first.wav_path,
                    offset_s=first.offset_s,
                    duration_s=group_duration,
                    src_text=" ".join(s.src_text.strip() for s in group).strip(),
                    tgt_text=" ".join(s.tgt_text.strip() for s in group).strip(),
                    speaker=group_speaker or last.speaker,
                    src_lang=first.src_lang,
                    tgt_lang=first.tgt_lang,
                )
            )
            group = []
            group_duration = 0.0
            group_speaker = None

        for seg in talk_segments:
            if not group:
                group = [seg]
                group_duration = float(seg.duration_s)
                group_speaker = seg.speaker
                continue

            # Ne fusionner que des segments de même speaker si possible (évite des enchaînements
            # incongrus). Si speaker est vide, on ignore la contrainte.
            if group_speaker and seg.speaker and seg.speaker != group_speaker:
                flush_group()
                group = [seg]
                group_duration = float(seg.duration_s)
                group_speaker = seg.speaker
                continue

            next_duration = group_duration + float(seg.duration_s)
            if next_duration > max_duration_s:
                # On flush avant d'ajouter le segment si on dépasserait la borne.
                flush_group()
                group = [seg]
                group_duration = float(seg.duration_s)
                group_speaker = seg.speaker
                continue

            # Ajouter le segment.
            group.append(seg)
            group_duration = next_duration

            # Critère de coupe : fin de phrase et durée suffisante.
            ends_sentence = _ends_with_sentence_punctuation(seg.src_text) or (
                _ends_with_sentence_punctuation(seg.tgt_text)
            )
            if ends_sentence and group_duration >= target_duration_s:
                flush_group()
                continue

            # Si on ne requiert pas la ponctuation, on peut flush proche de la cible.
            if not require_punctuation and group_duration >= target_duration_s:
                flush_group()

        flush_group()

    stats = {
        "segments_in": len(segments),
        "segments_out": len(merged),
        "merged_groups": merged_groups,
        "avg_merge_factor": (len(segments) / len(merged)) if merged else 0.0,
    }
    return merged, stats


def _load_soundfile():
    """Importer soundfile à la demande pour que les dry-run évitent la dépendance.

    Retour :
        Le module ``soundfile``.

    Lève :
        ImportError : Si soundfile n'est pas installé.
    """
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "soundfile is required for stage prepare. "
            "Install dependencies from requirements.txt."
        ) from exc
    return sf


def _resample_audio(data, src_rate: int, dst_rate: int):
    """Rééchantillonner un clip numpy flottant 1-D au taux cible.

    Utilise torchaudio si les taux diffèrent ; passe tel quel si identiques.

    Paramètres :
        data: Mono waveform samples (numpy array).
        src_rate: Sample rate read from the source FLAC.
        dst_rate: Target rate (typically 16000 Hz).

    Retour :
        Resampled numpy array at ``dst_rate``.

    Lève :
        ImportError : Si torch/torchaudio manquent et que le rééchantillonnage est requis.
    """
    if src_rate == dst_rate:
        return data
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "torchaudio is required for resampling when source rate != target rate."
        ) from exc
    # torchaudio.resample attend [canaux, temps].
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, src_rate, dst_rate)
    return resampled.squeeze(0).numpy()


def extract_and_save_wav(
    segment: SegmentRecord,
    destination: Path,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> int:
    """Extraire un segment YAML du FLAC vers WAV mono PCM16.

    Lit le FLAC parent, découpe par offset/durée en secondes, éventuellement
    rééchantillonne, écrit en PCM_16 à ``sample_rate``, et valide avant de retourner.

    Paramètres :
        segment: Source path, timing, and text metadata.
        destination: Output ``.wav`` path (parent dirs created as needed).
        sample_rate: Target sample rate (default 16000).

    Retour :
        Frame count at ``sample_rate`` after successful validation.

    Lève :
        ImportError: Missing soundfile or torchaudio (when resampling).
        RuntimeError : Validation post-écriture échouée (fichier partiel supprimé).
    """
    import numpy as np

    sf = _load_soundfile()
    data, sr = sf.read(segment.wav_path.as_posix(), always_2d=True)
    if data.size == 0:
        return 0
    # Mixer le FLAC multi-canal en mono avant découpage.
    mono = data.mean(axis=1)
    start_frame = int(segment.offset_s * sr)
    end_frame = start_frame + int(segment.duration_s * sr)
    clip = mono[start_frame:end_frame]
    if clip.size == 0:
        return 0
    if sr != sample_rate:
        clip = _resample_audio(clip, sr, sample_rate)
    # Borner à [-1, 1] avant quantification PCM_16.
    clip = np.clip(clip.astype("float32"), -1.0, 1.0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination.as_posix(), clip, sample_rate, subtype=TARGET_SUBTYPE)
    validation = validate_wav_file(destination, expected_sr=sample_rate)
    if not validation["ok"]:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"WAV validation failed for {destination.name}: "
            + "; ".join(validation["issues"])
        )
    return int(validation["n_frames"])


def validate_wav_file(
    path: Path,
    *,
    expected_sr: int = TARGET_SAMPLE_RATE,
    expected_frames: int | None = None,
) -> dict[str, Any]:
    """Vérifier qu'un WAV correspond aux attentes train/inférence aval.

    Pantagruel et SpeechBrain attendent du PCM16 mono 16 kHz ; vérifie métadonnées,
    sample integrity, peak range, optional frame count, and a torchaudio load.

    Paramètres :
        path: Path to the written segment WAV.
        expected_sr: Expected sample rate (default 16000).
        expected_frames: If set, ``n_frames`` must match (from manifest).

    Retour :
        Dict with keys ``ok``, ``n_frames``, ``peak``, and ``issues`` (str list).
    """
    import numpy as np

    issues: list[str] = []
    sf = _load_soundfile()
    if not path.is_file():
        return {"ok": False, "n_frames": 0, "issues": ["file_missing"]}

    info = sf.info(path.as_posix())
    if info.samplerate != expected_sr:
        issues.append(f"sample_rate={info.samplerate} expected={expected_sr}")
    if info.channels != TARGET_CHANNELS:
        issues.append(f"channels={info.channels} expected={TARGET_CHANNELS}")
    if info.subtype != TARGET_SUBTYPE:
        issues.append(f"subtype={info.subtype} expected={TARGET_SUBTYPE}")

    data, sr = sf.read(path.as_posix(), dtype="float32", always_2d=True)
    n_frames = int(data.shape[0])
    if sr != expected_sr:
        issues.append(f"read_sample_rate={sr}")
    if data.shape[1] != TARGET_CHANNELS:
        issues.append(f"read_channels={data.shape[1]}")
    if not np.isfinite(data).all():
        issues.append("non_finite_samples")
    peak = float(abs(data).max()) if data.size else 0.0
    if peak > 1.0:
        issues.append(f"peak_out_of_range={peak:.4f}")

    if expected_frames is not None and n_frames != expected_frames:
        issues.append(f"n_frames={n_frames} expected={expected_frames}")

    try:
        import torchaudio

        waveform, torch_sr = torchaudio.load(path.as_posix())
        if torch_sr != expected_sr:
            issues.append(f"torchaudio_sr={torch_sr}")
        if waveform.shape[0] != TARGET_CHANNELS:
            issues.append(f"torchaudio_channels={waveform.shape[0]}")
        if waveform.shape[1] != n_frames:
            issues.append(
                f"torchaudio_frames={waveform.shape[1]} soundfile_frames={n_frames}"
            )
    except ImportError:
        pass  # test fumée torchaudio optionnel (ex. env CI minimal)
    except OSError as exc:
        issues.append(f"torchaudio_load_failed:{exc}")

    return {"ok": not issues, "n_frames": n_frames, "peak": peak, "issues": issues}


def verify_prepared_outputs(
    *,
    langpair: str,
    output_root: Path,
    manifests_root: Path,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_errors: int = 20,
) -> tuple[list[str], dict[str, Any]]:
    """Vérifier que chaque ligne de manifest pointe vers un WAV valide sur disque.

    Résout les chemins ``audio`` relatifs à ``PROJECT_ROOT`` et réutilise
    ``validate_wav_file`` so stage 3+ never sees broken references.

    Paramètres :
        langpair: Language pair subdirectory under manifests.
        output_root: Processed audio root (for context; paths come from manifests).
        manifests_root: Directory containing ``<langpair>/*.tsv``.
        sample_rate: Expected WAV sample rate.
        max_errors: Stop collecting after this many issues (limits stderr noise).

    Retour :
        Tuple of (error messages, summary dict with per-split ok/invalid counts).
    """
    errors: list[str] = []
    summary: dict[str, Any] = {"splits": {}, "total_segments": 0, "invalid_segments": 0}

    for split in SPLITS:
        manifest_path = manifests_root / langpair / f"{split}.tsv"
        if not manifest_path.is_file():
            errors.append(f"Missing manifest: {manifest_path}")
            continue

        split_ok = 0
        split_bad = 0
        with manifest_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                summary["total_segments"] += 1
                audio_rel = row.get("audio", "")
                n_frames_str = row.get("n_frames", "")
                try:
                    expected_frames = int(n_frames_str)
                except ValueError:
                    expected_frames = None

                # Le manifest stocke des chemins POSIX relatifs au dépôt via manifest_audio_path().
                audio_path = PROJECT_ROOT / audio_rel
                if not audio_path.is_file():
                    split_bad += 1
                    errors.append(f"{split}/{row.get('id')}: missing audio {audio_rel}")
                else:
                    check = validate_wav_file(
                        audio_path,
                        expected_sr=sample_rate,
                        expected_frames=expected_frames,
                    )
                    if not check["ok"]:
                        split_bad += 1
                        errors.append(
                            f"{split}/{row.get('id')}: " + "; ".join(check["issues"])
                        )
                    else:
                        split_ok += 1

                if len(errors) >= max_errors:
                    break

        summary["splits"][split] = {"ok": split_ok, "invalid": split_bad}
        summary["invalid_segments"] += split_bad

    return errors, summary


def write_manifest_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Écrire un manifest TSV de split avec colonnes fixes et délimiteur tabulation.

    Paramètres :
        path: Destination ``<split>.tsv`` path.
        rows: Dict rows keyed by ``MANIFEST_COLUMNS``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_target_lines(path: Path, lines: list[str]) -> None:
    """Écrire une ligne cible normalisée par segment conservé (entrée SPM / BPE).

    Paramètres :
        path: Destination ``<split>.target.txt`` path.
        lines: Normalized target strings in manifest row order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def detect_leaks(
    *,
    ids_by_split: dict[str, set[str]],
    targets_by_split: dict[str, set[str]],
) -> list[str]:
    """Détecter le chevauchement d'ids utterance ou de texte cible entre splits (fuite train).

    Le PRD exige des splits train et dev/test disjoints ; des cibles normalisées dupliquées peuvent
    gonfler les métriques de validation même si les ids utterance diffèrent.

    Paramètres :
        ids_by_split: Utterance ids kept per split.
        targets_by_split: Normalized target strings kept per split.

    Retour :
        Chaînes d'issues lisibles (vide sans fuite).
    """
    issues: list[str] = []
    train_ids = ids_by_split.get("train", set())
    train_targets = targets_by_split.get("train", set())
    for split in ("valid", "test"):
        # Tout utterance train réutilisé en valid/test casse l'intégrité des splits.
        id_overlap = train_ids & ids_by_split.get(split, set())
        if id_overlap:
            issues.append(
                f"ID overlap train/{split}: {len(id_overlap)} utterance id(s)"
            )
        # Des lignes cible normalisées identiques entre splits peuvent fuiter le signal étiquette.
        tgt_overlap = train_targets & targets_by_split.get(split, set())
        if tgt_overlap:
            issues.append(
                f"Target text overlap train/{split}: {len(tgt_overlap)} line(s)"
            )
    return issues


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    """Écraser atomiquement l'instantané JSON reprise/progression pour les longs runs.

    Paramètres :
        path: Progress file (default ``artifacts/prepare_<langpair>.progress.json``).
        payload: Serializable counters for the current split.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _count_existing_wavs(pair_out: Path) -> int:
    """Compter les fichiers ``*.wav`` sous l'arborescence de sortie d'une paire (diag reprise).

    Paramètres :
        pair_out: ``<output-root>/<langpair>`` directory.

    Retour :
        Number of WAV files found recursively (0 if missing).
    """
    if not pair_out.is_dir():
        return 0
    return sum(1 for _ in pair_out.rglob("*.wav"))


def run_prepare(
    *,
    langpair: str,
    input_root: Path,
    output_root: Path,
    manifests_root: Path,
    sample_rate: int = 16000,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
    text_norm: str = "nfkc",
    lowercase: bool = False,
    segment_mode: str = "utterance",
    sentence_target_duration: float = 10.0,
    sentence_max_duration: float = 15.0,
    sentence_require_punctuation: bool = True,
    fail_on_leak: bool = True,
    dedupe_target_overlap: bool = False,
    resume: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    report_path: Path | None = None,
    progress_path: Path | None = None,
) -> tuple[PrepareReport, int]:
    """Préparer une paire de langues : segmenter audio, manifests, fuite et vérif WAV.

    Parcourt train/valid/test, filtre par durée et texte, extrait ou réutilise
    les WAV, écrit TSV + target.txt par split, puis exécute détection de fuite et vérification.

    Paramètres :
        langpair: Supported pair slug.
        input_root: Raw m-TEDx download root.
        output_root: Processed WAV output root.
        manifests_root: TSV and ``.target.txt`` destination.
        sample_rate: Target Hz (default 16000).
        min_duration: Drop segments shorter than this (seconds).
        max_duration: Drop segments longer than this (seconds).
        text_norm: ``nfkc`` or ``none``.
        lowercase: Lowercase normalized text when True.
        fail_on_leak: Map leakage to exit code 5 when True.
        resume: Reuse valid on-disk WAVs instead of re-extracting.
        dry_run: Count/filter only; skip I/O except corpus resolution.
        verbose: Print per-split and progress messages.
        report_path: JSON report destination.
        progress_path: Periodic progress JSON path.

    Retour :
        Tuple (``PrepareReport``, code de sortie pour ``run_from_namespace``).
    """
    report_path = report_path or (
        PROJECT_ROOT / "artifacts" / f"prepare_{langpair}.json"
    )
    progress_path = progress_path or (
        PROJECT_ROOT / "artifacts" / f"prepare_{langpair}.progress.json"
    )
    if dry_run:
        try:
            corpus = resolve_corpus_root(input_root, langpair)
        except FileNotFoundError:
            print(f"[dry-run] corpus missing under {input_root}; nothing to prepare")
            report = PrepareReport(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                langpair=langpair,
                input_root=str(input_root.resolve()),
                output_root=str(output_root.resolve()),
                manifests_root=str(manifests_root.resolve()),
                sample_rate=sample_rate,
                min_duration=min_duration,
                max_duration=max_duration,
                text_norm=text_norm,
                lowercase=lowercase,
                exit_code=0,
            )
            return report, 0
    else:
        corpus = resolve_corpus_root(input_root, langpair)

    pair_out = output_root / langpair
    pair_manifests = manifests_root / langpair
    if verbose and not dry_run:
        existing = _count_existing_wavs(pair_out)
        print(
            f"Prepare {langpair}: resume={'on' if resume else 'off'}, "
            f"existing_wavs={existing}"
        )
    split_stats: list[SplitStats] = []
    ids_by_split: dict[str, set[str]] = {}
    targets_by_split: dict[str, set[str]] = {}
    processing_errors: list[str] = []
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    target_lines_by_split: dict[str, list[str]] = {}
    sentence_like_stats: dict[str, Any] = {}

    for split in SPLITS:
        stats = SplitStats(split=split)
        rows: list[dict[str, Any]] = []
        target_lines: list[str] = []
        ids_by_split[split] = set()
        targets_by_split[split] = set()

        if not dry_run:
            split_dir = corpus / "data" / split
            if not split_dir.is_dir():
                processing_errors.append(f"Missing split directory: {split_dir}")
                split_stats.append(stats)
                continue

        try:
            segment_iter = iter_mtedx_segments(corpus, langpair, split)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            processing_errors.append(f"{split}: {exc}")
            split_stats.append(stats)
            continue

        segments = list(segment_iter)
        if segment_mode == "sentence_like":
            segments, split_sentence_stats = merge_segments_sentence_like(
                segments,
                target_duration_s=sentence_target_duration,
                max_duration_s=sentence_max_duration,
                require_punctuation=sentence_require_punctuation,
            )
            sentence_like_stats[split] = split_sentence_stats

        for segment in segments:
            stats.segments_in += 1
            src_text = normalize_text(
                segment.src_text, mode=text_norm, lowercase=lowercase
            )
            tgt_text = normalize_text(
                segment.tgt_text, mode=text_norm, lowercase=lowercase
            )

            if not src_text or not tgt_text:
                stats.segments_dropped += 1
                stats.drop_reasons["empty_text"] = (
                    stats.drop_reasons.get("empty_text", 0) + 1
                )
                continue

            duration = segment.duration_s
            if duration < min_duration:
                stats.segments_dropped += 1
                stats.drop_reasons["too_short"] = (
                    stats.drop_reasons.get("too_short", 0) + 1
                )
                continue
            if duration > max_duration:
                stats.segments_dropped += 1
                stats.drop_reasons["too_long"] = (
                    stats.drop_reasons.get("too_long", 0) + 1
                )
                continue

            if not dry_run and not segment.wav_path.is_file():
                stats.segments_dropped += 1
                stats.drop_reasons["missing_audio"] = (
                    stats.drop_reasons.get("missing_audio", 0) + 1
                )
                continue

            abs_audio = pair_out / split / f"{segment.utt_id}.wav"
            audio_field = manifest_audio_path(
                output_root, langpair, split, segment.utt_id
            )

            if dry_run:
                # Estimer le nombre de frames sans toucher FLAC/WAV (planification uniquement).
                n_frames = int(duration * sample_rate)
            else:
                reused = False
                # Reprise : ignorer la ré-extraction si un WAV segment existant est valide.
                if resume and abs_audio.is_file():
                    check = validate_wav_file(abs_audio, expected_sr=sample_rate)
                    if check["ok"]:
                        n_frames = int(check["n_frames"])
                        reused = True
                        stats.drop_reasons["resumed"] = (
                            stats.drop_reasons.get("resumed", 0) + 1
                        )
                    else:
                        # Cache corrompu ou mauvais format — supprimer et ré-extraire.
                        abs_audio.unlink(missing_ok=True)
                if not reused:
                    try:
                        n_frames = extract_and_save_wav(
                            segment, abs_audio, sample_rate=sample_rate
                        )
                    except (OSError, RuntimeError, ImportError) as exc:
                        stats.segments_dropped += 1
                        stats.drop_reasons["audio_error"] = (
                            stats.drop_reasons.get("audio_error", 0) + 1
                        )
                        processing_errors.append(f"{split}/{segment.utt_id}: {exc}")
                        continue

            if n_frames <= 0:
                stats.segments_dropped += 1
                stats.drop_reasons["empty_audio"] = (
                    stats.drop_reasons.get("empty_audio", 0) + 1
                )
                if not dry_run and abs_audio.exists():
                    abs_audio.unlink(missing_ok=True)
                continue

            stats.segments_kept += 1
            ids_by_split[split].add(segment.utt_id)
            targets_by_split[split].add(tgt_text)
            rows.append(
                {
                    "id": segment.utt_id,
                    "audio": audio_field,
                    "n_frames": n_frames,
                    "tgt_text": tgt_text,
                    "speaker": segment.speaker,
                    "tgt_lang": segment.tgt_lang,
                    "src_text": src_text,
                    "src_lang": segment.src_lang,
                }
            )
            target_lines.append(tgt_text)

            # Point de contrôle périodique pour prepares longs interrompus (tous les N segments).
            if not dry_run and stats.segments_in % PROGRESS_INTERVAL == 0:
                _write_progress(
                    progress_path,
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "langpair": langpair,
                        "split": split,
                        "resume": resume,
                        "segments_in": stats.segments_in,
                        "segments_kept": stats.segments_kept,
                        "segments_dropped": stats.segments_dropped,
                        "drop_reasons": stats.drop_reasons,
                        "wav_on_disk": _count_existing_wavs(pair_out),
                    },
                )
                if verbose:
                    print(
                        f"    progress {split}: in={stats.segments_in} "
                        f"kept={stats.segments_kept} "
                        f"resumed={stats.drop_reasons.get('resumed', 0)}"
                    )

        rows_by_split[split] = rows
        target_lines_by_split[split] = target_lines

        if verbose:
            print(
                f"  {split}: kept {stats.segments_kept}/{stats.segments_in} "
                f"(dropped {stats.segments_dropped})"
            )
        split_stats.append(stats)

    # Anti-fuite : train ne doit pas partager ids ou cibles normalisées avec valid/test.
    leak_issues = detect_leaks(
        ids_by_split=ids_by_split, targets_by_split=targets_by_split
    )

    if (
        dedupe_target_overlap
        and not dry_run
        and not processing_errors
        and leak_issues
        and "train" in targets_by_split
    ):
        train_targets = targets_by_split["train"]
        overlaps_valid = train_targets & targets_by_split.get("valid", set())
        overlaps_test = train_targets & targets_by_split.get("test", set())

        if verbose:
            total = len(overlaps_valid) + len(overlaps_test)
            print(
                f"  Dedupe target overlaps: {total} unique target(s) "
                "removed from valid/test."
            )

        for split, overlaps in (("valid", overlaps_valid), ("test", overlaps_test)):
            if not overlaps:
                continue
            original_rows = rows_by_split.get(split, [])
            kept_rows = [
                row for row in original_rows if row.get("tgt_text") not in overlaps
            ]
            removed = len(original_rows) - len(kept_rows)
            rows_by_split[split] = kept_rows
            target_lines_by_split[split] = [row["tgt_text"] for row in kept_rows]
            ids_by_split[split] = {row["id"] for row in kept_rows}
            targets_by_split[split] = {row["tgt_text"] for row in kept_rows}

            for stats in split_stats:
                if stats.split != split:
                    continue
                stats.segments_kept -= removed
                stats.segments_dropped += removed
                stats.drop_reasons["target_overlap_train"] = (
                    stats.drop_reasons.get("target_overlap_train", 0) + removed
                )

        leak_issues = detect_leaks(
            ids_by_split=ids_by_split, targets_by_split=targets_by_split
        )

    if not dry_run:
        for split in SPLITS:
            write_manifest_tsv(
                pair_manifests / f"{split}.tsv", rows_by_split.get(split, [])
            )
            write_target_lines(
                pair_manifests / f"{split}.target.txt",
                target_lines_by_split.get(split, []),
            )
    wav_errors: list[str] = []
    wav_summary: dict[str, Any] = {}
    if not dry_run and not processing_errors:
        wav_errors, wav_summary = verify_prepared_outputs(
            langpair=langpair,
            output_root=output_root,
            manifests_root=manifests_root,
            sample_rate=sample_rate,
        )
        if verbose and wav_summary:
            print(f"  WAV verification: {wav_summary}")

    # Priorité : traitement (4) conservé ; fuite (5) et vérif WAV (6) si encore 0.
    exit_code = 0
    if processing_errors:
        exit_code = 4
    if leak_issues and fail_on_leak:
        exit_code = 5 if exit_code == 0 else exit_code
    if wav_errors and exit_code == 0:
        exit_code = 6

    report = PrepareReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        langpair=langpair,
        input_root=str(input_root.resolve()),
        output_root=str(output_root.resolve()),
        manifests_root=str(manifests_root.resolve()),
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
        text_norm=text_norm,
        lowercase=lowercase,
        segment_mode=segment_mode,
        sentence_like={
            "target_duration_s": sentence_target_duration,
            "max_duration_s": sentence_max_duration,
            "require_punctuation": sentence_require_punctuation,
            "stats": sentence_like_stats,
        }
        if segment_mode == "sentence_like"
        else {},
        splits=split_stats,
        leak_issues=leak_issues,
        exit_code=exit_code,
    )

    if not dry_run:
        _write_report(
            report_path,
            report,
            processing_errors,
            wav_errors=wav_errors,
            wav_summary=wav_summary,
        )
        if processing_errors:
            print(
                f"\nPrepare FAILED ({len(processing_errors)} error(s)).",
                file=sys.stderr,
            )
            for item in processing_errors[:20]:
                print(f"  - {item}", file=sys.stderr)
        elif leak_issues and fail_on_leak:
            print("\nPrepare FAILED: data leakage detected.", file=sys.stderr)
            for item in leak_issues:
                print(f"  - {item}", file=sys.stderr)
        elif wav_errors:
            print(
                f"\nPrepare FAILED: WAV verification ({len(wav_errors)} issue(s)).",
                file=sys.stderr,
            )
            for item in wav_errors[:20]:
                print(f"  - {item}", file=sys.stderr)
        else:
            print(f"\nPrepare complete. Report: {report_path}")
            print(
                f"  WAV format: {TARGET_SAMPLE_RATE} Hz, mono, {TARGET_SUBTYPE} "
                f"({wav_summary.get('total_segments', 0)} segments checked)"
            )
            if leak_issues:
                print(
                    f"  Warning: {len(leak_issues)} leak check(s) "
                    "(--no-fail-on-leak allowed continuation)",
                    file=sys.stderr,
                )

    return report, exit_code


def _write_report(
    path: Path,
    report: PrepareReport,
    processing_errors: list[str],
    *,
    wav_errors: list[str] | None = None,
    wav_summary: dict[str, Any] | None = None,
) -> None:
    """Persister le rapport prepare plus erreurs optionnelles et détails de validation WAV.

    Paramètres :
        path: JSON report path.
        report: Core ``PrepareReport`` dataclass.
        processing_errors: Per-segment or split-level failures.
        wav_errors: Manifest/WAV verification messages.
        wav_summary: Aggregate counts from ``verify_prepared_outputs``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload["processing_errors"] = processing_errors
    payload["wav_validation_errors"] = wav_errors or []
    payload["wav_validation_summary"] = wav_summary or {}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construire le parseur d'arguments CLI de l'étape 2.

    Retour :
        Configured ``ArgumentParser`` for ``2_prepare.py`` / pipeline prepare.
    """
    parser = argparse.ArgumentParser(
        description="S3T Étape 2 — Préparer audio et manifests m-TEDx",
    )
    parser.add_argument("--langpair", required=True, help="e.g. fr-en")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "processed",
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--text-norm", choices=("nfkc", "none"), default="nfkc")
    parser.add_argument("--lowercase", action="store_true", default=False)
    parser.add_argument(
        "--segment-mode",
        choices=SEGMENT_MODES,
        default="utterance",
        help=(
            "Mode de segmentation intra-fichier. "
            "'utterance' = segments m-TEDx natifs ; "
            "'sentence_like' = fusion contiguë pour approcher des phrases complètes."
        ),
    )
    parser.add_argument(
        "--sentence-target-duration",
        type=float,
        default=10.0,
        help="Durée cible (s) des segments fusionnés en mode sentence_like.",
    )
    parser.add_argument(
        "--sentence-max-duration",
        type=float,
        default=15.0,
        help="Durée max (s) des segments fusionnés en mode sentence_like.",
    )
    parser.add_argument(
        "--sentence-require-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "En mode sentence_like, couper préférentiellement sur ponctuation forte "
            "(.?!). Désactiver pour couper dès la durée cible atteinte."
        ),
    )
    parser.add_argument(
        "--fail-on-leak",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dedupe-target-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Nettoyer la fuite : retirer de valid/test les segments dont la cible normalisée "
            "apparaît aussi dans train."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip segments whose output WAV already exists and is valid (default: on)",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=None,
        help="Progress JSON path (default: artifacts/prepare_<langpair>.progress.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="JSON report path (default: artifacts/prepare_<langpair>.json)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip processing; only verify existing manifests and WAV files",
    )
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par ``main()`` et la sous-commande prepare de ``scripts_communs/pipeline.py``.

    Paramètres :
        args: Parsed namespace from ``build_parser()``.

    Retour :
        Code de sortie processus (0 succès ; 2, 4, 5, 6 comme dans la doc du module).
    """
    try:
        langpair = parse_langpair(args.langpair)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    report_path = args.report or (
        PROJECT_ROOT / "artifacts" / f"prepare_{langpair}.json"
    )

    if args.verify_only:
        wav_errors, wav_summary = verify_prepared_outputs(
            langpair=langpair,
            output_root=args.output_root,
            manifests_root=args.manifests_root,
            sample_rate=args.sample_rate,
        )
        report = PrepareReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            langpair=langpair,
            input_root=str(args.input_root.resolve()),
            output_root=str(args.output_root.resolve()),
            manifests_root=str(args.manifests_root.resolve()),
            sample_rate=args.sample_rate,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            text_norm=args.text_norm,
            lowercase=args.lowercase,
            exit_code=0 if not wav_errors else 6,
        )
        _write_report(
            report_path,
            report,
            [],
            wav_errors=wav_errors,
            wav_summary=wav_summary,
        )
        if wav_errors:
            print(
                f"WAV verification FAILED ({len(wav_errors)} issue(s)).",
                file=sys.stderr,
            )
            for item in wav_errors[:20]:
                print(f"  - {item}", file=sys.stderr)
            return 6
        print(
            f"WAV verification OK — {wav_summary.get('total_segments', 0)} segment(s)."
        )
        print(f"Report: {report_path}")
        return 0

    try:
        _, exit_code = run_prepare(
            langpair=langpair,
            input_root=args.input_root,
            output_root=args.output_root,
            manifests_root=args.manifests_root,
            sample_rate=args.sample_rate,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            text_norm=args.text_norm,
            lowercase=args.lowercase,
            segment_mode=str(getattr(args, "segment_mode", "utterance")),
            sentence_target_duration=float(
                getattr(args, "sentence_target_duration", 10.0)
            ),
            sentence_max_duration=float(getattr(args, "sentence_max_duration", 15.0)),
            sentence_require_punctuation=bool(
                getattr(args, "sentence_require_punctuation", True)
            ),
            fail_on_leak=args.fail_on_leak,
            dedupe_target_overlap=getattr(args, "dedupe_target_overlap", False),
            resume=args.resume,
            dry_run=args.dry_run,
            verbose=args.verbose,
            report_path=report_path,
            progress_path=args.progress,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Entrée CLI : analyser les arguments et déléguer à ``run_from_namespace``.

    Paramètres :
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Retour :
        Code de sortie de ``run_from_namespace``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
