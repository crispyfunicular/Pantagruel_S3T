#!/usr/bin/env bash
# OVH — Waiter : attend la fin de run_chain_055_052 puis enchaîne run_061 → run_062.
#
# Chaîne suivante (~22–29 h GPU cumulées) :
#   run_061 ST L-114k SPM 5k + gel 15k (encodeur gated — OVH)
#   run_062 speechLLM Mistral-7B 4-bit (VRAM ~14 Go — impraticable IMAG 11 Go)
#
# Usage (OVH, pendant run_chain_055_052) :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_wait_chain_055_052_done_then_061_062.sh \
#     > logs/run_chain_061_062_waiter.log 2>&1 &
#   tail -f logs/run_chain_061_062_waiter.log
#
# Arrêter le waiter sans lancer la chaîne :
#   pkill -f run_ovh_wait_chain_055_052_done_then_061_062.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
CHAIN_SCRIPT="${ROOT}/scripts/run_ovh_chain_061_st_114k_freeze15k_then_062_mistral.sh"

gpu_or_chain_busy() {
  pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1 \
    || pgrep -af "run_ovh_chain_055_speechllm_llama_seed2_then_052" >/dev/null 2>&1
}

echo "=== $(date -Is) Waiter 061→062 — attente fin chaîne 055→052 (poll ${POLL_SEC}s) ==="
while gpu_or_chain_busy; do
  pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
  pgrep -af "run_ovh_chain_055" | head -1 || true
  echo "$(date -Is) GPU ou chaîne 055→052 encore active — prochaine vérif dans ${POLL_SEC}s"
  sleep "$POLL_SEC"
done

echo "=== $(date -Is) Chaîne 055→052 terminée — lancement run_chain_061_062 ==="
exec bash "$CHAIN_SCRIPT"
