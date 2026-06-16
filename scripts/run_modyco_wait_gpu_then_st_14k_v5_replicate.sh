#!/usr/bin/env bash
# Modyco — attend GPU disponible (serveur partagé) puis lance run_043 (réplication run_026).
#
# Critères GPU libre : aucun pipeline.py train/run ET VRAM utilisée < 4 Go
# (évite collision avec ollama ou autre job tiers sur la tour partagée).
#
# Usage (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_wait_gpu_then_st_14k_v5_replicate.sh \
#     > logs/run_043_modyco_wait_chain.log 2>&1 &
#   tail -f logs/run_043_modyco_wait_chain.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

wait_gpu_available() {
  echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s, VRAM max ${VRAM_MAX_MIB} MiB) ==="
  while true; do
    local vram pipeline
    vram="$(gpu_vram_used_mib || echo 99999)"
    if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
      pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
      echo "$(date -Is) pipeline S3T actif — attente ${POLL_SEC}s"
    elif [[ "${vram}" =~ ^[0-9]+$ ]] && (( vram > VRAM_MAX_MIB )); then
      echo "$(date -Is) VRAM occupée (${vram} MiB > ${VRAM_MAX_MIB}) — attente ${POLL_SEC}s"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -3 || true
    else
      echo "=== $(date -Is) GPU libre (VRAM ${vram} MiB) ==="
      return 0
    fi
    sleep "$POLL_SEC"
  done
}

wait_gpu_available
echo "=== $(date -Is) Lancement run_043 (réplication run_026 v5 SpecAugment) ==="
exec bash "${ROOT}/scripts/run_modyco_st_14k_v5_replicate.sh"
