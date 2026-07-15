#!/usr/bin/env bash
# Modyco — Waiter GPU puis run_055 Llama seed 2 (piste F).
#
# Usage :
#   bash scripts/run_modyco_wait_gpu_then_speechllm_055_llama_seed2.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POLL_SEC="${POLL_SEC:-300}"
VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"
MIN_FREE_MIB="${MIN_FREE_MIB:-28000}"
TARGET="${ROOT}/scripts/run_modyco_speechllm_14k_llama32_3b_seed2.sh"

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

echo "=== $(date -Is) Waiter run_055 Llama seed2 — poll ${POLL_SEC}s (free ≥ ${MIN_FREE_MIB} MiB) ==="
while ! gpu_usable; do
  echo "$(date -Is) VRAM used=$(gpu_vram_used_mib) free=$(gpu_vram_free_mib) MiB — attente"
  sleep "$POLL_SEC"
done

chmod +x "$TARGET"
exec bash "$TARGET"
