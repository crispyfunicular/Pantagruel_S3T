#!/usr/bin/env bash
# OVH — après run_032/033 : ST L-114k v9 SpecAugment temporel + fréquentiel (run_038).
#
# Chaîne cumulée : attendre la fin de run_ovh_wait_chain_post_v6_speechllm_then_st_spm5k.sh
# puis lancer run_038 (~9–12 h GPU).
#
# Usage (OVH) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_ovh_wait_chain_post_032_033_st_specaug_freq.sh \
#     > logs/run_038_ovh_wait_chain.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"

wait_chain_032_033_done() {
  local chain_pid
  chain_pid="$(pgrep -f "run_ovh_wait_chain_post_v6_speechllm_then_st_spm5k.sh" | head -1 || true)"
  if [[ -z "$chain_pid" ]]; then
    echo "=== $(date -Is) Chaîne run_032/033 déjà terminée (ou jamais lancée) ==="
    return 0
  fi
  echo "=== $(date -Is) Attente fin chaîne run_032/033 (PID ${chain_pid}) ==="
  while kill -0 "$chain_pid" 2>/dev/null; do
    echo "$(date -Is) Chaîne run_032/033 active — nouvelle vérif dans ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
  echo "=== $(date -Is) Chaîne run_032/033 terminée ==="
}

wait_gpu_free() {
  echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s) ==="
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
    echo "$(date -Is) GPU occupée — nouvelle vérif dans ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

wait_chain_032_033_done
wait_gpu_free
echo "=== $(date -Is) Lancement run_038 ST L-114k v9 SpecAugment freq ==="
bash "${ROOT}/scripts/run_ovh_st_114k_v9_specaug_freq.sh"

echo "=== $(date -Is) Chaîne OVH post-032/033 (run_038) terminée ==="
