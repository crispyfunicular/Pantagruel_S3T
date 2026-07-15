#!/usr/bin/env bash
# Modyco — Waiter : attend GPU libre puis lance run_074 Canary (max 15 h).
#
# Préparé pour session utilisateur (~15 h max). NE PAS LANCER sans feu vert.
#
# Usage (Modyco) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_modyco_wait_gpu_then_open_canary_074.sh \
#     > logs/run_waiter_canary_074_modyco.log 2>&1 &
#   tail -f logs/run_waiter_canary_074_modyco.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POLL_SEC="${POLL_SEC:-300}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-6}"
CANARY_SCRIPT="${ROOT}/scripts/run_modyco_open_canary_074.sh"

gpu_busy() {
  pgrep -af "^python.*pipeline\.py (train|run|evaluate)" >/dev/null 2>&1
}

echo "=== $(date -Is) Waiter Canary Modyco — attente GPU libre (poll ${POLL_SEC}s, max ${MAX_WAIT_HOURS}h) ==="
deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
while gpu_busy; do
  if [[ "$(date +%s)" -ge "$deadline" ]]; then
    echo "=== $(date -Is) TIMEOUT attente GPU (${MAX_WAIT_HOURS}h) ===" >&2
    exit 3
  fi
  pgrep -af "^python.*pipeline\.py (train|run|evaluate)" | head -1 || true
  echo "$(date -Is) GPU encore occupé — prochaine vérif dans ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

echo "=== $(date -Is) GPU libre — lancement Canary run_074 (Modyco) ==="
chmod +x "$CANARY_SCRIPT"
exec bash "$CANARY_SCRIPT"
