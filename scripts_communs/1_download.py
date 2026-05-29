#!/usr/bin/env python3
"""
Étape 1 — Télécharger les corpus m-TEDx depuis OpenSLR-100.

Récupère les archives ``.tgz`` par paire de langues, les extrait optionnellement sous
``datasets/raw/``, et écrit un manifeste JSON pour la reproductibilité. Prend en charge
la reprise HTTP Range pour les gros téléchargements.

Paire de langues par défaut : fr-en uniquement (focus phase 1 S3T).
Politique d'intégrité : basique (téléchargement + extraction ; pas encore de sommes de contrôle).

Sorties :
    - ``datasets/raw/mtedx_<langpair>.tgz``
    - Arborescence extraite avec ``data/train`` (voir ``resolve_extracted_corpus``)
    - ``artifacts/download_manifest.json``

Codes de sortie : 0 succès, 2 mauvaise paire CLI, 4 une ou plusieurs erreurs de téléchargement.

Usage :
    python scripts_communs/1_download.py
    python scripts_communs/1_download.py --langpairs fr-en,fr-pt
    python scripts_communs/1_download.py --dry-run
    python scripts_communs/1_download.py --no-extract
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OPENSRLR_BASE = "https://www.openslr.org/resources/100"

LANGPAIR_ARCHIVES: dict[str, str] = {
    "fr-en": f"{OPENSRLR_BASE}/mtedx_fr-en.tgz",
    "fr-pt": f"{OPENSRLR_BASE}/mtedx_fr-pt.tgz",
    "fr-es": f"{OPENSRLR_BASE}/mtedx_fr-es.tgz",
}

DEFAULT_LANGPAIRS = ("fr-en",)

CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass
class DownloadResult:
    """Résultat téléchargement/extraction par paire de langues."""

    langpair: str
    url: str
    archive_path: str
    extracted_dir: str | None = None
    status: str = "ok"
    message: str = ""


@dataclass
class DownloadReport:
    """Résumé écrit dans ``download_manifest.json``."""

    timestamp_utc: str
    output_root: str
    langpairs: list[str]
    resume: bool
    extract: bool
    results: list[DownloadResult] = field(default_factory=list)
    exit_code: int = 0


def parse_langpairs(value: str) -> list[str]:
    """Analyser les paires de langues séparées par des virgules et valider les archives connues."""
    pairs = [p.strip() for p in value.split(",") if p.strip()]
    if not pairs:
        raise ValueError("At least one language pair is required in --langpairs")
    unknown = [p for p in pairs if p not in LANGPAIR_ARCHIVES]
    if unknown:
        supported = ", ".join(sorted(LANGPAIR_ARCHIVES))
        raise ValueError(f"Unknown langpair(s): {unknown}. Supported: {supported}")
    return pairs


def _archive_path(output_root: Path, langpair: str) -> Path:
    """Chemin local de l'archive ``.tgz`` téléchargée."""
    return output_root / f"mtedx_{langpair}.tgz"


def _extracted_marker(output_root: Path, langpair: str) -> Path:
    """Nom de répertoire corpus préféré après extraction (peut différer ; voir le résolveur)."""
    return output_root / f"mtedx_{langpair}"


def resolve_extracted_corpus(output_root: Path, langpair: str) -> Path | None:
    """Retourner le répertoire m-TEDx extrait si le split train est présent."""
    for name in (f"mtedx_{langpair}", langpair):
        path = output_root / name
        if (path / "data" / "train").is_dir():
            return path
    return None


def download_file(
    url: str,
    destination: Path,
    *,
    resume: bool = True,
    verbose: bool = False,
) -> None:
    """Télécharger un fichier avec reprise HTTP Range optionnelle."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() and resume else 0

    headers: dict[str, str] = {}
    mode = "wb"
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        mode = "ab"

    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=60) as response:
            status = getattr(response, "status", 200)
            if existing_size > 0 and status == 416:
                if verbose:
                    print(f"  Archive already complete: {destination}")
                return
            if existing_size > 0 and status not in (206, 200):
                destination.unlink(missing_ok=True)
                existing_size = 0
                mode = "wb"
                request = Request(url, method="GET")
                with urlopen(request, timeout=60) as full_response:
                    _write_stream(full_response, destination, mode, verbose)
                return
            _write_stream(response, destination, mode, verbose)
    except HTTPError as exc:
        if existing_size > 0 and exc.code == 416:
            if verbose:
                print(f"  Archive already complete: {destination}")
            return
        raise


def _write_stream(response, destination: Path, mode: str, verbose: bool) -> None:
    """Écrire le corps de la réponse HTTP sur disque par blocs de 1 MiB."""
    total = response.headers.get("Content-Length")
    total_int = int(total) if total and total.isdigit() else None
    downloaded = 0
    with destination.open(mode) as handle:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if verbose and total_int:
                pct = min(100, int(100 * downloaded / total_int))
                print(f"\r  Downloaded {downloaded}/{total_int} bytes ({pct}%)", end="")
    if verbose:
        print(f"\n  Saved: {destination}")


def extract_archive(
    archive: Path, output_root: Path, langpair: str, verbose: bool
) -> Path:
    """Extraire l'archive .tgz dans output_root."""
    existing = resolve_extracted_corpus(output_root, langpair)
    if existing is not None:
        if verbose:
            print(f"  Corpus already extracted: {existing}")
        return existing

    if verbose:
        print(f"  Extracting {archive} -> {output_root}")
    with tarfile.open(archive, "r:gz") as tar:
        # Python 3.12+ exige un filtre explicite pour une extraction plus sûre.
        tar.extractall(path=output_root, filter="data")

    resolved = resolve_extracted_corpus(output_root, langpair)
    if resolved is None:
        raise tarfile.TarError(
            f"Could not find data/train after extracting {archive} into {output_root}"
        )
    if verbose:
        print(f"  Extracted corpus root: {resolved}")
    return resolved


def run_download(
    *,
    langpairs: list[str],
    output_root: Path,
    resume: bool = True,
    extract: bool = True,
    dry_run: bool = False,
    verbose: bool = False,
    manifest_path: Path | None = None,
) -> tuple[DownloadReport, int]:
    """Télécharger et éventuellement extraire les archives m-TEDx."""
    manifest_path = manifest_path or (
        PROJECT_ROOT / "artifacts" / "download_manifest.json"
    )
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    results: list[DownloadResult] = []

    for langpair in langpairs:
        url = LANGPAIR_ARCHIVES[langpair]
        archive = _archive_path(output_root, langpair)

        if dry_run:
            print(f"[dry-run] would download {langpair}: {url} -> {archive}")
            if extract:
                marker = resolve_extracted_corpus(output_root, langpair)
                target = marker or _extracted_marker(output_root, langpair)
                print(f"[dry-run] would extract -> {target}")
            results.append(
                DownloadResult(
                    langpair=langpair,
                    url=url,
                    archive_path=str(archive),
                    extracted_dir=str(
                        resolve_extracted_corpus(output_root, langpair)
                        or _extracted_marker(output_root, langpair)
                    )
                    if extract
                    else None,
                    status="dry_run",
                    message="Skipped (dry-run)",
                )
            )
            continue

        try:
            if verbose:
                print(f"Downloading {langpair} from {url}")
            download_file(url, archive, resume=resume, verbose=verbose)

            extracted_dir: Path | None = None
            if extract:
                extracted_dir = extract_archive(archive, output_root, langpair, verbose)

            results.append(
                DownloadResult(
                    langpair=langpair,
                    url=url,
                    archive_path=str(archive),
                    extracted_dir=str(extracted_dir) if extracted_dir else None,
                    status="ok",
                    message="Download complete",
                )
            )
        except (OSError, URLError, HTTPError, tarfile.TarError) as exc:
            results.append(
                DownloadResult(
                    langpair=langpair,
                    url=url,
                    archive_path=str(archive),
                    status="error",
                    message=str(exc),
                )
            )

    failed = [r for r in results if r.status == "error"]
    exit_code = 0 if not failed else 4

    report = DownloadReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        output_root=str(output_root.resolve()),
        langpairs=langpairs,
        resume=resume,
        extract=extract,
        results=results,
        exit_code=exit_code,
    )

    if not dry_run:
        _write_manifest(manifest_path, report)
        if failed:
            print(f"\nDownload FAILED ({len(failed)} error(s)).", file=sys.stderr)
            for item in failed:
                print(f"  - {item.langpair}: {item.message}", file=sys.stderr)
        else:
            print(f"\nDownload complete. Manifest: {manifest_path}")

    return report, exit_code


def _write_manifest(path: Path, report: DownloadReport) -> None:
    """Persister ``DownloadReport`` en JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": report.timestamp_utc,
        "output_root": report.output_root,
        "langpairs": report.langpairs,
        "resume": report.resume,
        "extract": report.extract,
        "results": [asdict(r) for r in report.results],
        "exit_code": report.exit_code,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI pour l'étape 1."""
    parser = argparse.ArgumentParser(
        description="S3T Étape 1 — Téléchargement m-TEDx (OpenSLR-100)",
    )
    parser.add_argument(
        "--langpairs",
        default=",".join(DEFAULT_LANGPAIRS),
        help="Paires de langues séparées par des virgules (défaut : fr-en)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "raw",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reprendre les téléchargements partiels si possible (défaut : activé)",
    )
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extraire le .tgz après téléchargement (défaut : activé)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "download_manifest.json",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """Point d'entrée utilisé par ``pipeline.py download``."""
    try:
        langpairs = parse_langpairs(args.langpairs)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    _, exit_code = run_download(
        langpairs=langpairs,
        output_root=args.output_root,
        resume=args.resume,
        extract=args.extract,
        dry_run=args.dry_run,
        verbose=args.verbose,
        manifest_path=args.manifest,
    )
    return exit_code


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI principal."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_namespace(args)


if __name__ == "__main__":
    sys.exit(main())
