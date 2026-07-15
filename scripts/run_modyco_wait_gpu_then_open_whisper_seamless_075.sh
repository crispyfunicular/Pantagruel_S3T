#!/usr/bin/env bash
# Modyco — Waiter : attend GPU utilisable puis Whisper-ST + SeamlessM4T v2 (piste K).
#
# Après run_074 Canary (terminé). Attend :
#   - aucun pipeline S3T train/run/evaluate ;
#   - VRAM < 4 Go OU ≥ 18 Go libres (tour partagée, processus tiers possibles).
#
# Usage (Modyco) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_modyco_wait_gpu_then_open_whisper_seamless_075.sh \
#     > logs/run_waiter_open_075_modyco.log 2>&1 &
#   tail -f logs/run_waiter_open_075_modyco.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POLL_SEC="${POLL_SEC:-300}"
VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"
TARGET_SCRIPT="${ROOT}/scripts/run_modyco_open_whisper_seamless_075.sh"

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_vram_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_usable() {
  local vram_used vram_free
  vram_used="$(gpu_vram_used_mib || echo 99999)"
  vram_free="$(gpu_vram_free_mib || echo 0)"
  if pgrep -af "^python.*pipeline\.py (train|run|evaluate)" >/dev/null 2>&1; then
    return 1
  fi
  if [[ "${vram_used}" =~ ^[0-9]+$ ]] && (( vram_used <= VRAM_MAX_MIB )); then
    return 0
  fi
  if [[ "${vram_free}" =~ ^[0-9]+$ ]] && (( vram_free >= MIN_FREE_MIB )); then
    return 0
  fi
  return 1
}

wait_gpu_usable() {
  echo "=== $(date -Is) Waiter open 075 Modyco — poll ${POLL_SEC}s (VRAM max ${VRAM_MAX_MIB} MiB ou free ≥ ${MIN_FREE_MIB} MiB) ==="
  while ! gpu_usable; do
    if pgrep -af "^python.*pipeline\.py (train|run|evaluate)" >/dev/null 2>&1; then
      pgrep -af "^python.*pipeline\.py (train|run|evaluate)" | head -1 || true
      echo "$(date -Is) pipeline S3T actif — attente ${POLL_SEC}s"
    else
      echo "$(date -Is) VRAM used=$(gpu_vram_used_mib) MiB free=$(gpu_vram_free_mib) MiB — attente ${POLL_SEC}s"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -3 || true
    fi
    sleep "$POLL_SEC"
  done
  echo "=== $(date -Is) GPU utilisable — lancement Whisper + Seamless (Modyco) ==="
}

wait_gpu_usable
chmod +x "$TARGET_SCRIPT"
exec bash "$TARGET_SCRIPT"
