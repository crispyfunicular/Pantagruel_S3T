#!/usr/bin/env python3
"""Retire OVH / modyco / aker des docs et pages web (prose, libellés, chemins affichés)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    ROOT / "docs",
    ROOT / "documentation",
    ROOT / "README.md",
    ROOT / "rapport.md",
]

# Ordre important : expressions longues d'abord. Ne pas utiliser \\baker\\b (casse aker.sh).
REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bIMAG\s*\(\s*aker\s*\)", re.I), "cluster GETALP"),
    (re.compile(r"\bIMAG\s+aker\b", re.I), "cluster GETALP"),
    (re.compile(r"\bIMAG\s*/\s*aker\b", re.I), "cluster GETALP"),
    (re.compile(r"\b\(aker\)", re.I), "(cluster GETALP)"),
    (re.compile(r"\baker\.imag\.fr\b", re.I), "<serveur-login-cluster>"),
    (re.compile(r"\bligone\.imag\.fr\b", re.I), "<bastion-cluster>"),
    (re.compile(r"\blig-gpu\d+\.imag\.fr\b", re.I), "<nœud-gpu>"),
    (re.compile(r"\blig-gpu\d+\b", re.I), "nœud GPU"),
    (re.compile(r"\blig-gpu\b", re.I), "nœud GPU"),
    (re.compile(r"\bModyco\s*\(\s*tour\s*\)", re.I), "machine GPU locale"),
    (re.compile(r"\bModyco\b"), "machine GPU locale"),
    (re.compile(r"\bmodyco\b"), "machine GPU locale"),
    (re.compile(r"\bOVH\b"), "serveur cloud GPU"),
    (re.compile(r"\bovh\b"), "serveur cloud GPU"),
    (re.compile(r"\bAker\b"), "cluster GETALP"),
    (re.compile(r"(?<![a-zA-Z-])IMAG(?![a-zA-Z-])"), "cluster GETALP"),
    # Scripts affichés dans la doc (noms de fichiers)
    (re.compile(r"`run_modyco_[^`]+`"), "`script d'orchestration (machine GPU locale)`"),
    (re.compile(r"`run_ovh_[^`]+`"), "`script d'orchestration (serveur cloud GPU)`"),
    (re.compile(r"`run_aker_[^`]+`"), "`script d'orchestration (cluster GETALP)`"),
    (re.compile(r"`run_oar_[^`]*_aker[^`]*`"), "`script OAR (cluster GETALP)`"),
    (re.compile(r"`run_oar_[^`]+`"), "`script OAR (cluster GETALP)`"),
    (re.compile(r"\[`run_modyco_[^\]]+`\]\([^)]+\)"), "`scripts/` (machine GPU locale)"),
    (re.compile(r"\[`run_ovh_[^\]]+`\]\([^)]+\)"), "`scripts/` (serveur cloud GPU)"),
    (re.compile(r"\[`scripts/run_modyco_[^\]]+`\]\([^)]+\)"),
     "`scripts/` (machine GPU locale)"),
    (re.compile(r"\[`scripts/run_ovh_[^\]]+`\]\([^)]+\)"),
     "`scripts/` (serveur cloud GPU)"),
    (re.compile(r"\[`scripts/run_aker_[^\]]+`\]\([^)]+\)"),
     "`scripts/` (cluster GETALP)"),
    (re.compile(r"\[`scripts/run_oar_[^\]]+`\]\([^)]+\)"),
     "`scripts/` (cluster GETALP, OAR)"),
    (re.compile(r"`scripts/run_modyco_[^`]+`"), "`scripts/` (machine GPU locale)"),
    (re.compile(r"`scripts/run_ovh_[^`]+`"), "`scripts/` (serveur cloud GPU)"),
    (re.compile(r"`scripts/run_aker_[^`]+`"), "`scripts/` (cluster GETALP)"),
    (re.compile(r"`scripts/run_oar_[^`]+`"), "`scripts/` (cluster GETALP, OAR)"),
    (re.compile(r"`scripts/aker\.sh`"), "`scripts/` (déploiement cluster)"),
    (re.compile(r"\[`scripts/run_aker_smoke\.sh`\]\([^)]+\)"),
     "`scripts/` (smoke test cluster)"),
    (re.compile(r"`scripts/aker\.ssh\.config\.example`"),
     "`scripts/*.ssh.config.example`"),
    (re.compile(r"scripts/run_modyco_\*\.sh"), "scripts d'orchestration (machine GPU locale)"),
    (re.compile(r"scripts/run_ovh_\*\.sh"), "scripts d'orchestration (serveur cloud GPU)"),
    (re.compile(r"run_modyco_\*\.sh"), "scripts d'orchestration (machine GPU locale)"),
    (re.compile(r"run_ovh_\*\.sh"), "scripts d'orchestration (serveur cloud GPU)"),
    (re.compile(r"bash scripts/run_modyco_[^\s\\]+"), "bash scripts/<orchestration-machine-locale>"),
    (re.compile(r"bash scripts/run_ovh_[^\s\\]+"), "bash scripts/<orchestration-serveur-cloud>"),
    (re.compile(r"bash scripts/run_aker_[^\s\\]+"), "bash scripts/<orchestration-cluster>"),
    (re.compile(r"nohup bash scripts/run_ovh_[^\s]+"), "nohup bash scripts/<orchestration-serveur-cloud>"),
    (re.compile(r"nohup bash scripts/run_modyco_[^\s\\]+"),
     "nohup bash scripts/<orchestration-machine-locale>"),
    (re.compile(r"~/S3T/scripts/run_oar_[^\s']+"), "~/S3T/scripts/<script-oar-cluster>"),
    (re.compile(r"/home/getalp/bonapelm/S3T/scripts/run_oar_[^\s]+"),
     "/home/.../S3T/scripts/<script-oar-cluster>"),
    (re.compile(r"\./scripts/cluster GETALP\.sh"), "./scripts/<deploy-cluster>.sh"),
    (re.compile(r"scripts/cluster GETALP\.ssh\.config\.example"),
     "scripts/*.ssh.config.example"),
    (re.compile(r"cluster GETALP\.sh"), "<deploy-cluster>.sh"),
    (re.compile(r"bonapelm@bastion du cluster"), "utilisateur@<bastion-cluster>"),
    (re.compile(r"bonapelm@serveur du cluster"), "utilisateur@<serveur-login-cluster>"),
    (re.compile(r"nœud GPU\.imag\.fr"), "<nœud-gpu>"),
    (re.compile(r"GPU_HOST=nœud GPU\.imag\.fr"), "GPU_HOST=<nœud-gpu>"),
    # Libellés tableaux
    (re.compile(r"Canary-1B\s*\(\s*machine GPU locale\s*\)", re.I), "Canary-1B (réplication)"),
    (re.compile(r"SeamlessM4T v2\s*\(\s*serveur cloud GPU\s*\)", re.I), "SeamlessM4T v2"),
    (re.compile(r"Whisper-ST\s*\(\s*serveur cloud GPU\s*\)", re.I), "Whisper-ST"),
    (re.compile(r"Open ST \*\*SeamlessM4T v2 cluster GETALP\*\*"),
     "Open ST **SeamlessM4T v2 (cluster)**"),
    (re.compile(r"Open ST \*\*Whisper cluster GETALP\*\*"),
     "Open ST **Whisper (cluster)**"),
    (re.compile(r"Open ST \*\*Canary-1B machine GPU locale\*\*"),
     "Open ST **Canary-1B (réplication)**"),
    (re.compile(r"Open ST \*\*Whisper machine GPU locale\*\*"),
     "Open ST **Whisper (réplication)**"),
    (re.compile(r"Open ST \*\*Seamless machine GPU locale\*\*"),
     "Open ST **SeamlessM4T v2 (réplication)**"),
    (re.compile(r"speechLLM B1 \*\*L-14k cluster GETALP repl\.\*\*"),
     "speechLLM B1 **L-14k (cluster, réplication)**"),
    (re.compile(r"speechLLM B1 \*\*L-14k cluster GETALP couche (\d+)\*\*"),
     r"speechLLM B1 **L-14k (cluster, couche \1)**"),
    (re.compile(r"## Serveur cluster GETALP \(déploiement pipelines\)", re.I),
     "## Déploiement sur cluster GPU (GETALP)"),
    (re.compile(r"§ Serveur cluster GETALP"), "§ Déploiement sur cluster GPU"),
]

RUN_MACHINE = re.compile(
    r"(?<=[a-z0-9])_(?:ovh|modyco|aker)(?=_|$)",
    re.I,
)
RUN_SHORT = re.compile(
    r"\b(run_\d+(?:[a-z0-9_]*))_(?:ovh|modyco|aker)\b",
    re.I,
)


def scrub_run_ids(text: str) -> str:
    text = RUN_SHORT.sub(r"\1", text)
    return RUN_MACHINE.sub("", text)


def scrub_file(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    text = raw
    for pat, repl in REPLACEMENTS:
        text = pat.sub(repl, text)
    text = scrub_run_ids(text)
    if text != raw:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed: list[str] = []
    for target in TARGETS:
        paths = [target] if target.is_file() else list(target.rglob("*"))
        for path in paths:
            if path.suffix.lower() not in {".html", ".md", ".txt"}:
                continue
            if path.name.startswith("_"):
                continue
            if scrub_file(path):
                changed.append(str(path.relative_to(ROOT)))
    print(f"Fichiers modifiés : {len(changed)}")
    for p in sorted(changed):
        print(f"  - {p}")


if __name__ == "__main__":
    main()
