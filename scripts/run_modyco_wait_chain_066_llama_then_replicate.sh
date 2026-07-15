#!/usr/bin/env bash
# Modyco — Chaîne waiters : run_066 Llama dégelé → run_023 Phi-2 replicate (piste L + H).
#
# 1. Attend GPU utilisable (Llama : ≥ 24 Go libres ou VRAM quasi vide)
# 2. run_066_modyco Llama unfreeze (~6–12 h)
# 3. Attend GPU libre (Phi-2 : pipeline train/run absent)
# 4. run_023 speechLLM L-14k replicate (~2 h)
#
# Usage (Modyco) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_modyco_wait_chain_066_llama_then_replicate.sh \
#     > logs/run_waiter_chain_066_replicate_modyco.log 2>&1 &
#   tail -f logs/run_waiter_chain_066_replicate_modyco.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POLL_SEC="${POLL_SEC:-300}"

wait_pipeline_idle() {
  echo "=== $(date -Is) Attente fin pipeline train/run (poll ${POLL_SEC}s) ==="
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
    sleep "$POLL_SEC"
  done
}

echo "=== $(date -Is) CHAÎNE Modyco 066→023 — étape 1/2 : waiter Llama ==="
chmod +x \
  "${ROOT}/scripts/run_modyco_wait_gpu_then_speechllm_066_llama_unfreeze.sh" \
  "${ROOT}/scripts/run_modyco_speechllm_066_llama_unfreeze.sh" \
  "${ROOT}/scripts/run_modyco_speechllm_14k_replicate.sh"
bash "${ROOT}/scripts/run_modyco_wait_gpu_then_speechllm_066_llama_unfreeze.sh"

wait_pipeline_idle

echo "=== $(date -Is) CHAÎNE Modyco — étape 2/2 : speechLLM Phi-2 replicate (run_023) ==="
bash "${ROOT}/scripts/run_modyco_speechllm_14k_replicate.sh"

echo "=== $(date -Is) CHAÎNE 066→023 TERMINÉE ==="
