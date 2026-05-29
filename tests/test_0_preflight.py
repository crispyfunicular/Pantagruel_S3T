"""Tests for scripts_communs/0_preflight.py."""

from __future__ import annotations

import json
from pathlib import Path

from conftest import PROJECT_ROOT, load_stage_module

preflight = load_stage_module("0_preflight.py")


def test_python_version_check_passes_on_current_interpreter():
    result = preflight._check_python_version()
    assert result.status == "pass"
    assert result.critical is True


def test_dry_run_returns_zero_and_skips_report_write(tmp_path: Path):
    output = tmp_path / "preflight_report.json"
    _, exit_code = preflight.run_preflight(
        min_disk_gb=1,
        min_vram_gb=1,
        check_gpu=False,
        check_network=False,
        output=output,
        dry_run=True,
    )
    assert exit_code == 0
    assert not output.exists()


def test_report_written_with_expected_keys(tmp_path: Path):
    output = tmp_path / "preflight_report.json"
    _, exit_code = preflight.run_preflight(
        min_disk_gb=1,
        min_vram_gb=1,
        check_gpu=False,
        check_network=False,
        output=output,
        disk_path=PROJECT_ROOT,
    )
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "checks" in data
    assert "summary" in data
    assert "timestamp_utc" in data
    assert "exit_code" in data
    assert isinstance(data["summary"]["failed_critical"], int)
    # exit code follows strict_critical policy
    assert exit_code == (0 if data["summary"]["passed"] else 1)
