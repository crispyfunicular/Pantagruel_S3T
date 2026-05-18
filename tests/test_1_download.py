"""Tests for scripts/1_download.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from conftest import PROJECT_ROOT, load_stage_module

download = load_stage_module("1_download.py")


def test_parse_langpairs_default_supported():
    assert download.parse_langpairs("fr-en") == ["fr-en"]


def test_parse_langpairs_rejects_unknown():
    try:
        download.parse_langpairs("fr-xx")
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "Unknown langpair" in str(exc)


def test_dry_run_writes_no_files(tmp_path: Path):
    output_root = tmp_path / "raw"
    manifest = tmp_path / "manifest.json"
    _, exit_code = download.run_download(
        langpairs=["fr-en"],
        output_root=output_root,
        dry_run=True,
        manifest_path=manifest,
    )
    assert exit_code == 0
    assert not output_root.exists()
    assert not manifest.exists()


def test_run_download_success_mocked(tmp_path: Path):
    output_root = tmp_path / "raw"
    manifest = tmp_path / "manifest.json"

    with (
        patch.object(download, "download_file") as mock_dl,
        patch.object(download, "extract_archive") as mock_extract,
    ):
        mock_extract.return_value = output_root / "mtedx_fr-en"
        _, exit_code = download.run_download(
            langpairs=["fr-en"],
            output_root=output_root,
            extract=True,
            manifest_path=manifest,
        )

    assert exit_code == 0
    mock_dl.assert_called_once()
    mock_extract.assert_called_once()
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["langpairs"] == ["fr-en"]
    assert data["results"][0]["status"] == "ok"


def test_build_parser_default_langpairs():
    parser = download.build_parser()
    args = parser.parse_args([])
    assert args.langpairs == "fr-en"
    assert args.output_root == PROJECT_ROOT / "datasets" / "raw"
