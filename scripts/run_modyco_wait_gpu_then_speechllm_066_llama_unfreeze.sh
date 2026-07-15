#!/usr/bin/env bash
# Modyco — Waiter : attend GPU utilisable puis run_066 Llama dégelé (piste L).
#
# Critères : aucun pipeline train/run ; VRAM < 4 Go OU ≥ 24 Go libres (Llama + encodeur).
#
# Usage (Modyco) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_modyco_wait_gpu_then_speechllm_066_llama_unfreeze.sh \
#     > logs/run_waiter_066_modyco.log 2>&1 &
#   tail -f logs/run_waiter_066_modyco.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POLL_SEC="${POLL_SEC:-300}"
VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"
MIN_FREE_MIB="${MIN_FREE_MIB:-28000}"
TARGET_SCRIPT="${ROOT}/scripts/run_modyco_speechllm_066_llama_unfreeze.sh"

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_vram_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_usable_for_train() {
  local vram_used vram_free
  vram_used="$(gpu_vram_used_mib || echo 99999)"
  vram_free="$(gpu_vram_free_mib || echo 0)"
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
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
  echo "=== $(date -Is) Waiter run_066 Llama Modyco — poll ${POLL_SEC}s (VRAM ≤${VRAM_MAX_MIB} MiB ou free ≥${MIN_FREE_MIB} MiB) ==="
  while ! gpu_usable_for_train; do
    if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
      pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
      echo "$(date -Is) pipeline S3T actif — attente ${POLL_SEC}s"
    else
      echo "$(date -Is) VRAM used=$(gpu_vram_used_mib) MiB free=$(gpu_vram_free_mib) MiB — attente ${POLL_SEC}s"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -3 || true
    fi
    sleep "$POLL_SEC"
  done
  echo "=== $(date -Is) GPU utilisable — lancement run_066 Llama dégelé ==="
}

wait_gpu_usable
chmod +x "$TARGET_SCRIPT"
exec bash "$TARGET_SCRIPT"
