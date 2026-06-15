#!/usr/bin/env bash
# Modyco — attend run_026 puis lance run_027 ST L-14k v6 long (120k + SpecAugment).
#
# Usage (Modyco, session détachable) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_wait_st_then_st_14k_v6_long.sh \
#     > logs/run_027_modyco_wait_chain.log 2>&1 &

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

echo "=== $(date -Is) GPU libre — lancement run_027 ST L-14k v6 long ==="
exec bash "${ROOT}/scripts/run_modyco_st_14k_v6_long.sh"
