#!/usr/bin/env bash
# OVH — Waiter : attend la fin de run_062 (éval incluse) puis lance run_063 ST.
#
# Usage (pendant run_062) :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_wait_062_done_then_063.sh \
#     > logs/run_waiter_063.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
ST_SCRIPT="${ROOT}/scripts/run_ovh_st_063_l14k_specaug_strong.sh"

gpu_or_chain_busy() {
  pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1 \
    || pgrep -af "run_ovh_speechllm_062_mistral" >/dev/null 2>&1
}

echo "=== $(date -Is) Waiter run_063 — attente fin run_062 (poll ${POLL_SEC}s) ==="
while gpu_or_chain_busy; do
  pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
  echo "$(date -Is) GPU ou run_062 encore actif — prochaine vérif dans ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

echo "=== $(date -Is) run_062 terminé — lancement run_063 ==="
exec bash "$ST_SCRIPT"
