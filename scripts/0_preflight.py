#!/usr/bin/env python3
"""
Stage 0 — Preflight checks for S3T pipeline (Linux + CUDA target).

Policy: strict_critical — exit non-zero only if a critical check fails.
Warnings do not block the pipeline.

Usage:
    python scripts/0_preflight.py
    python scripts/0_preflight.py --check-gpu --min-disk-gb 200 --min-vram-gb 8
    python scripts/0_preflight.py --output artifacts/preflight_report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NETWORK_URLS = (
    "https://www.openslr.org/",
    "https://huggingface.co/",
)

CheckStatus = Literal["pass", "fail", "warn", "skip"]


@dataclass
class CheckResult:
    name: str
    status: CheckStatus
    critical: bool
    observed: Any = None
    required: Any = None
    message: str = ""


@dataclass
class PreflightReport:
    timestamp_utc: str
    host: str
    platform: str
    python_version: str
    project_root: str
    input_thresholds: dict[str, Any] = field(default_factory=dict)
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


def _check_python_version(min_major: int = 3, min_minor: int = 10) -> CheckResult:
    version = sys.version_info
    ok = (version.major, version.minor) >= (min_major, min_minor)
    observed = f"{version.major}.{version.minor}.{version.micro}"
    required = f">= {min_major}.{min_minor}"
    return CheckResult(
        name="python_version",
        status="pass" if ok else "fail",
        critical=True,
        observed=observed,
        required=required,
        message="Python version OK" if ok else f"Python {required} required",
    )


def _check_torch_import() -> CheckResult:
    try:
        import torch  # noqa: F401

        version = torch.__version__
        return CheckResult(
            name="torch_import",
            status="pass",
            critical=True,
            observed=version,
            required="importable",
            message=f"torch {version} importable",
        )
    except ImportError as exc:
        return CheckResult(
            name="torch_import",
            status="fail",
            critical=True,
            observed=None,
            required="importable",
            message=f"torch not importable: {exc}",
        )


def _check_cuda_available(check_gpu: bool) -> CheckResult:
    if not check_gpu:
        return CheckResult(
            name="cuda_available",
            status="skip",
            critical=False,
            message="GPU check disabled (--no-check-gpu)",
        )
    try:
        import torch
    except ImportError:
        return CheckResult(
            name="cuda_available",
            status="fail",
            critical=True,
            observed=False,
            required=True,
            message="Cannot check CUDA: torch not installed",
        )

    available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if available else None
    return CheckResult(
        name="cuda_available",
        status="pass" if available else "fail",
        critical=True,
        observed={"available": available, "device": device_name},
        required=True,
        message=f"CUDA available: {device_name}" if available else "CUDA not available (required for remote GPU run)",
    )


def _check_disk_space(path: Path, min_gb: int) -> CheckResult:
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        ok = free_gb >= min_gb
        return CheckResult(
            name="disk_space",
            status="pass" if ok else "fail",
            critical=True,
            observed=round(free_gb, 2),
            required=min_gb,
            message=f"Free disk at {path}: {free_gb:.1f} GB (need >= {min_gb} GB)",
        )
    except OSError as exc:
        return CheckResult(
            name="disk_space",
            status="fail",
            critical=True,
            observed=None,
            required=min_gb,
            message=f"Cannot read disk usage for {path}: {exc}",
        )


def _query_vram_gb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        line = out.stdout.strip().splitlines()[0]
        # nvidia-smi reports MiB
        return float(line.strip()) / 1024.0
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def _check_vram(min_gb: int) -> CheckResult:
    vram_gb = _query_vram_gb()
    if vram_gb is None:
        return CheckResult(
            name="vram",
            status="warn",
            critical=False,
            observed=None,
            required=min_gb,
            message="Could not query VRAM (nvidia-smi missing or failed)",
        )
    ok = vram_gb >= min_gb
    return CheckResult(
        name="vram",
        status="pass" if ok else "warn",
        critical=False,
        observed=round(vram_gb, 2),
        required=min_gb,
        message=f"GPU VRAM: {vram_gb:.1f} GB (recommended >= {min_gb} GB)",
    )


def _check_nvidia_smi() -> CheckResult:
    path = shutil.which("nvidia-smi")
    ok = path is not None
    return CheckResult(
        name="nvidia_smi",
        status="pass" if ok else "warn",
        critical=False,
        observed=path,
        required="on PATH",
        message=f"nvidia-smi found: {path}" if ok else "nvidia-smi not found on PATH",
    )


def _check_url(url: str, timeout_s: float) -> bool:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 400
    except URLError:
        try:
            with urlopen(url, timeout=timeout_s) as resp:
                return 200 <= resp.status < 400
        except URLError:
            return False
    except (TimeoutError, socket.timeout, OSError):
        return False


def _check_network(check_network: bool, timeout_s: float) -> list[CheckResult]:
    if not check_network:
        return [
            CheckResult(
                name="network",
                status="skip",
                critical=False,
                message="Network check disabled (--no-check-network)",
            )
        ]
    results: list[CheckResult] = []
    for url in NETWORK_URLS:
        ok = _check_url(url, timeout_s)
        host = url.split("/")[2]
        results.append(
            CheckResult(
                name=f"network_{host}",
                status="pass" if ok else "warn",
                critical=False,
                observed=ok,
                required=True,
                message=f"Reachable: {url}" if ok else f"Unreachable: {url}",
            )
        )
    return results


def _check_path_exists(path: Path, name: str) -> CheckResult:
    exists = path.exists()
    return CheckResult(
        name=name,
        status="pass" if exists else "warn",
        critical=False,
        observed=str(path),
        required="exists",
        message=f"{name} present" if exists else f"{name} missing: {path}",
    )


def run_preflight(
    *,
    min_disk_gb: int = 200,
    min_vram_gb: int = 8,
    check_gpu: bool = True,
    check_network: bool = True,
    output: Path | None = None,
    disk_path: Path | None = None,
    network_timeout_s: float = 10.0,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[PreflightReport, int]:
    disk_path = disk_path or PROJECT_ROOT
    output = output or (PROJECT_ROOT / "artifacts" / "preflight_report.json")

    checks: list[CheckResult] = [
        _check_python_version(),
        _check_torch_import(),
        _check_cuda_available(check_gpu),
        _check_disk_space(disk_path, min_disk_gb),
        _check_vram(min_vram_gb),
        _check_nvidia_smi(),
        *_check_network(check_network, network_timeout_s),
        _check_path_exists(PROJECT_ROOT / "datasets", "datasets_dir"),
        _check_path_exists(PROJECT_ROOT / "scripts", "scripts_dir"),
    ]

    failed_critical = sum(1 for c in checks if c.critical and c.status == "fail")
    warning_count = sum(1 for c in checks if c.status == "warn")
    passed = failed_critical == 0
    exit_code = 0 if passed else 1

    report = PreflightReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        host=socket.gethostname(),
        platform=platform.platform(),
        python_version=platform.python_version(),
        project_root=str(PROJECT_ROOT),
        input_thresholds={
            "min_disk_gb": min_disk_gb,
            "min_vram_gb": min_vram_gb,
            "check_gpu": check_gpu,
            "check_network": check_network,
            "disk_path": str(disk_path),
        },
        checks=checks,
        summary={
            "passed": passed,
            "failed_critical": failed_critical,
            "warning_count": warning_count,
            "total_checks": len(checks),
        },
        exit_code=exit_code,
    )

    if dry_run:
        print("[dry-run] preflight checks would run:")
        for c in checks:
            print(f"  - {c.name}: {c.status} (critical={c.critical})")
        print(f"[dry-run] report would be written to: {output}")
        return report, 0

    _print_report(checks, verbose=verbose)
    _write_json_report(output, report)
    print(f"\nReport written to: {output}")
    if passed:
        print("Preflight PASSED (no critical failures).")
    else:
        print(f"Preflight FAILED ({failed_critical} critical failure(s)).", file=sys.stderr)

    return report, exit_code


def _print_report(checks: list[CheckResult], verbose: bool = False) -> None:
    icons = {"pass": "OK", "fail": "FAIL", "warn": "WARN", "skip": "SKIP"}
    print("\nPreflight checks:")
    for c in checks:
        label = icons.get(c.status, c.status.upper())
        line = f"  [{label:4}] {c.name}: {c.message}"
        print(line)
        if verbose and c.observed is not None:
            print(f"         observed={c.observed!r} required={c.required!r}")


def _write_json_report(path: Path, report: PreflightReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": report.timestamp_utc,
        "host": report.host,
        "platform": report.platform,
        "python_version": report.python_version,
        "project_root": report.project_root,
        "input_thresholds": report.input_thresholds,
        "checks": [asdict(c) for c in report.checks],
        "summary": report.summary,
        "exit_code": report.exit_code,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="S3T Stage 0 — Preflight (Linux + CUDA, strict_critical)",
    )
    parser.add_argument("--min-disk-gb", type=int, default=200)
    parser.add_argument("--min-vram-gb", type=int, default=8)
    parser.add_argument(
        "--check-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require CUDA availability (default: enabled).",
    )
    parser.add_argument(
        "--check-network",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Probe OpenSLR and Hugging Face (default: enabled).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "preflight_report.json",
    )
    parser.add_argument(
        "--disk-path",
        type=Path,
        default=PROJECT_ROOT,
        help="Path used for free-disk check (default: project root).",
    )
    parser.add_argument("--network-timeout", type=float, default=10.0)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    _, exit_code = run_preflight(
        min_disk_gb=args.min_disk_gb,
        min_vram_gb=args.min_vram_gb,
        check_gpu=args.check_gpu,
        check_network=args.check_network,
        output=args.output,
        disk_path=args.disk_path,
        network_timeout_s=args.network_timeout,
        dry_run=getattr(args, "dry_run", False),
        verbose=getattr(args, "verbose", False),
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
