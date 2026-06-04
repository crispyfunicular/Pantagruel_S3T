"""Tests manifest corpus externe (oralité)."""

from __future__ import annotations

from pathlib import Path

from scripts_communs.external_corpus import (
    discover_wav_lab_pairs,
    read_french_lab,
    write_external_manifest,
)


def test_read_french_lab(tmp_path: Path) -> None:
    """Lecture d'un fichier .lab."""
    lab = tmp_path / "1-1.lab"
    lab.write_text("Il fait beau aujourd'hui.\n", encoding="utf-8")
    assert read_french_lab(lab) == "Il fait beau aujourd'hui."


def test_discover_and_write_manifest(tmp_path: Path) -> None:
    """Découverte WAV+.lab et écriture TSV."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "1-1.lab").write_text("Phrase un.", encoding="utf-8")
    (corpus / "2-1.lab").write_text("Phrase deux.", encoding="utf-8")
    # Pas de vrai WAV : n_frames=0 ; read_manifest fonctionne quand même pour les tests.
    (corpus / "1-1.wav").write_bytes(b"\x00" * 100)
    (corpus / "2-1.wav").write_bytes(b"\x00" * 100)

    items = discover_wav_lab_pairs(corpus)
    assert len(items) == 2
    assert items[0].sample_id == "1-1"
    assert items[0].src_text == "Phrase un."

    out = tmp_path / "manifest.tsv"
    write_external_manifest(items, out)
    text = out.read_text(encoding="utf-8")
    assert "src_text" in text
    assert "Phrase un." in text
    assert "Phrase un." in text
    lines = [ln for ln in text.splitlines() if ln.startswith("1-1\t")]
    assert len(lines) == 1
    assert lines[0].split("\t")[3] == ""  # tgt_text vide
