"""Tests utilitaires variante 5 (train.target.txt)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MULTIMODAL_ROOT = PROJECT_ROOT / "5_Pantagruel_multimodal"
if str(MULTIMODAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MULTIMODAL_ROOT))

from multimodal_common import ensure_train_target_text  # noqa: E402


def test_ensure_train_target_text_from_tsv(tmp_path: Path) -> None:
    """Générer train.target.txt depuis un TSV minimal."""
    manifest = tmp_path / "train.tsv"
    target = tmp_path / "train.target.txt"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "tgt_text"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({"id": "a", "tgt_text": "Hello world."})
        writer.writerow({"id": "b", "tgt_text": "Second line."})

    assert ensure_train_target_text(target, train_manifest=manifest) == 0
    lines = target.read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["Hello world.", "Second line."]
