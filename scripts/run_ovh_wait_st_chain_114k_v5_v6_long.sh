#!/usr/bin/env bash
# OVH — attend run_025 puis enchaîne run_028 v5 SpecAugment + run_030 v6 long (120k).
#
# Chaîne longue (~25–30 h GPU cumulées) : v5 (80k) puis v6 long (120k).
#
# Usage (OVH) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_ovh_wait_st_chain_114k_v5_v6_long.sh \
#     > logs/run_028_030_ovh_wait_chain.log 2>&1 &

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
echo "=== $(date -Is) Lancement run_028 ST L-114k v5 SpecAugment ==="
bash "${ROOT}/scripts/run_ovh_st_114k_v5.sh"

wait_gpu_free
echo "=== $(date -Is) Lancement run_030 ST L-114k v6 long ==="
bash "${ROOT}/scripts/run_ovh_st_114k_v6_long.sh"

echo "=== $(date -Is) Chaîne OVH v5 + v6 long terminée ==="
