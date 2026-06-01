"""Tests CLI variante 5 — délégation vers 1_Transformer (dry-run)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "5_Pantagruel_multimodal" / "configs" / "fr-en" / "base.yaml"


def test_multimodal_train_dry_run() -> None:
    """``train --dry-run`` délègue sans erreur."""
    proc = subprocess.run(
        [
            sys.executable,
            "5_Pantagruel_multimodal/pipeline.py",
            "train",
            "--config",
            str(CONFIG),
            "--run-id",
            "run_test_multimodal",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "train_multimodal" in proc.stdout or "train stage plan" in proc.stdout


def test_multimodal_spm_dry_run() -> None:
    """``spm --dry-run`` planifie SentencePiece sentence_like."""
    proc = subprocess.run(
        [
            sys.executable,
            "5_Pantagruel_multimodal/pipeline.py",
            "spm",
            "--config",
            str(CONFIG),
            "--dry-run",
            "--overwrite",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "SentencePiece" in proc.stdout or "dry-run" in proc.stdout
