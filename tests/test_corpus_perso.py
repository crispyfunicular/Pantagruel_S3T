"""Tests corpus personnel (manifest + parsing référence EN)."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts_communs.corpus_perso import (
    build_corpus_perso_items,
    parse_reference_paragraphs,
    wav_id_to_paragraph_index,
    write_corpus_perso_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus_perso"
REFERENCE = CORPUS_DIR / "corpus_perso_ref_EN.txt"


@pytest.mark.skipif(not REFERENCE.is_file(), reason="corpus_perso absent")
def test_parse_reference_paragraphs_count() -> None:
    """200 phrases → 40 paragraphes de 5."""
    paragraphs = parse_reference_paragraphs(REFERENCE)
    assert len(paragraphs) == 40
    assert all(len(p.sentences) == 5 for p in paragraphs)
    assert paragraphs[0].text.startswith("He will protect himself")


def test_wav_id_to_paragraph_index() -> None:
    """Mapping N-K → index de paragraphe."""
    assert wav_id_to_paragraph_index("1-1") == 0
    assert wav_id_to_paragraph_index("1-2") == 1
    assert wav_id_to_paragraph_index("20-2") == 39


@pytest.mark.skipif(not CORPUS_DIR.is_dir(), reason="corpus_perso absent")
def test_build_corpus_perso_items_integration(tmp_path: Path) -> None:
    """40 WAV + références → manifest cohérent."""
    items = build_corpus_perso_items(
        corpus_dir=CORPUS_DIR,
        reference_path=REFERENCE,
        project_root=PROJECT_ROOT,
    )
    assert len(items) == 40
    assert items[0].sample_id == "1-1"
    assert "He will protect himself" in items[0].tgt_text
    assert items[1].sample_id == "1-2"
    assert "My father gave me permission" in items[1].tgt_text

    out = tmp_path / "manifest.tsv"
    write_corpus_perso_manifest(items, out)
    text = out.read_text(encoding="utf-8")
    assert "tgt_text" in text
    assert "corpus_perso/1-1.wav" in text
