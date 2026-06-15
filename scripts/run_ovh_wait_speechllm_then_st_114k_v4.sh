#!/usr/bin/env bash
# OVH — attend la fin du job GPU courant (ex. run_022 speechLLM) puis lance run_025 ST L-114k v4.
#
# Usage (sur OVH, session détachable) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_ovh_wait_speechllm_then_st_114k_v4.sh \
#     > logs/run_025_ovh_wait_chain.log 2>&1 &
#   tail -f logs/run_025_ovh_wait_chain.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"

echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s) ==="
while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
  pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
  echo "$(date -Is) GPU occupée — nouvelle vérif dans ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

echo "=== $(date -Is) GPU libre — lancement run_025 ST L-114k v4 ==="
exec bash "${ROOT}/scripts/run_ovh_st_114k_v4.sh"
