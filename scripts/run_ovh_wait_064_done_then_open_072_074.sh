#!/usr/bin/env bash
# OVH — Waiter : attend la fin de run_064 (ou tout train/run GPU) puis enchaîne
# run_072 → run_073 → run_074 (open baselines, variante 6).
#
# Usage (OVH, pendant run_064 ou autre entraînement) :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_wait_064_done_then_open_072_074.sh \
#     > logs/run_waiter_open_072_074.log 2>&1 &
#   tail -f logs/run_waiter_open_072_074.log
#
# Arrêter le waiter sans lancer la chaîne :
#   pkill -f run_ovh_wait_064_done_then_open_072_074.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
CHAIN_SCRIPT="${ROOT}/scripts/run_ovh_chain_open_072_then_073_then_074.sh"

gpu_or_chain_busy() {
  pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1 \
    || pgrep -af "run_ovh_st_064" >/dev/null 2>&1 \
    || pgrep -af "run_064_transformer_baseline_utterance_large_114k" >/dev/null 2>&1
}

echo "=== $(date -Is) Waiter open 072→074 — attente GPU libre (poll ${POLL_SEC}s) ==="
while gpu_or_chain_busy; do
  pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
  pgrep -af "run_ovh_st_064" | head -1 || true
  echo "$(date -Is) GPU ou run_064 encore actif — prochaine vérif dans ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

echo "=== $(date -Is) GPU libre — lancement chaîne open baselines 072→073→074 ==="
exec bash "$CHAIN_SCRIPT"
