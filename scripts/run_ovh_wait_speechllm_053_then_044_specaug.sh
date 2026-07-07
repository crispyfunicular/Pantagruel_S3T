#!/usr/bin/env bash
# OVH — attend la fin de run_053 (speechLLM L-114k + Llama) puis lance run_044 (L-114k SpecAugment).
#
# Usage (sur OVH, pendant ou après run_053) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_ovh_wait_speechllm_053_then_044_specaug.sh \
#     > logs/run_044_ovh_wait_chain.log 2>&1 &
#   tail -f logs/run_044_ovh_wait_chain.log
#
# Arrêter le waiter sans lancer run_044 :
#   pkill -f run_ovh_wait_speechllm_053_then_044_specaug.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"

echo "=== $(date -Is) Attente fin run_053 (poll ${POLL_SEC}s) ==="
while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
  pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
  echo "$(date -Is) GPU occupée (run_053 ou autre) — nouvelle vérif dans ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

echo "=== $(date -Is) GPU libre — lancement run_044 speechLLM L-114k SpecAugment ==="
exec bash "${ROOT}/scripts/run_ovh_speechllm_114k_v5_specaug.sh"
