#!/usr/bin/env bash
# OVH — après run_030 v6 long : speechLLM L-114k replicate puis ST L-114k v7 SPM 5k.
#
# Chaîne (~15–18 h GPU cumulées) : run_032 (~4–6 h) puis run_033 (~10–12 h).
# À lancer pendant ou après la fin de run_ovh_wait_st_chain_114k_v5_v6_long.sh.
#
# Usage (OVH) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_ovh_wait_chain_post_v6_speechllm_then_st_spm5k.sh \
#     > logs/run_032_033_ovh_wait_chain.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"

wait_gpu_free() {
  echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s) ==="
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
    echo "$(date -Is) GPU occupée — nouvelle vérif dans ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

wait_gpu_free
echo "=== $(date -Is) Lancement run_032 speechLLM L-114k replicate (48 tok) ==="
bash "${ROOT}/scripts/run_ovh_speechllm_114k_replicate.sh"

wait_gpu_free
echo "=== $(date -Is) Lancement run_033 ST L-114k v7 SPM 5k SpecAugment ==="
bash "${ROOT}/scripts/run_ovh_st_114k_v7_spm5k.sh"

echo "=== $(date -Is) Chaîne OVH post-v6 (speechLLM + ST SPM 5k) terminée ==="
