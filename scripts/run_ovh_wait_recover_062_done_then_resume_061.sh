#!/usr/bin/env bash
# OVH — Waiter : attend la fin de la reprise eval 052/061 + run_062, puis reprend run_061 (--resume).
#
# File GPU (5 juil. 2026) :
#   1. run_ovh_recover_eval_052_061_then_062_mistral.sh  (en cours)
#   2. run_061 ST L-114k --resume  (ce waiter, ~8–10 h GPU, max 20 h)
#
# Usage (OVH, pendant la reprise 052/061/062) :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_wait_recover_062_done_then_resume_061.sh \
#     > logs/run_waiter_resume_061.log 2>&1 &
#   tail -f logs/run_waiter_resume_061.log
#
# Arrêter le waiter sans lancer la reprise :
#   pkill -f run_ovh_wait_recover_062_done_then_resume_061.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
RESUME_SCRIPT="${ROOT}/scripts/run_ovh_resume_061_st_114k_freeze15k.sh"

gpu_or_chain_busy() {
  pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1 \
    || pgrep -af "run_ovh_recover_eval_052_061_then_062_mistral" >/dev/null 2>&1
}

echo "=== $(date -Is) Waiter resume run_061 — attente fin reprise 052/061/062 (poll ${POLL_SEC}s) ==="
while gpu_or_chain_busy; do
  pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
  pgrep -af "run_ovh_recover_eval_052_061" | head -1 || true
  echo "$(date -Is) GPU ou reprise 052/061/062 encore active — prochaine vérif dans ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

echo "=== $(date -Is) Reprise 052/061/062 terminée — lancement run_061 --resume ==="
exec bash "$RESUME_SCRIPT"
