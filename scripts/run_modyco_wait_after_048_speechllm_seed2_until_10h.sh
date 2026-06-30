#!/usr/bin/env bash
# Modyco — après run_048 (couche 6) : run_050 speechLLM seed 2 si la fenêtre le permet.
#
# Deadline : GPU libre avant DEADLINE_LOCAL (défaut 2026-06-23 10:00:00).
# Le run_050 est estimé ~2,5–6 h (plafond 8 h) : on ne lance que si ≥ MIN_HOURS restantes.
#
# Usage (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_wait_after_048_speechllm_seed2_until_10h.sh \
#     > logs/chain_048_050_modyco_wait.log 2>&1 &
#   tail -f logs/chain_048_050_modyco_wait.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
DEADLINE_LOCAL="${DEADLINE_LOCAL:-2026-06-23 10:00:00}"
MIN_HOURS_FOR_NEXT="${MIN_HOURS_FOR_NEXT:-6}"
MAX_RUN_HOURS="${MAX_RUN_HOURS:-8}"

deadline_epoch() {
  date -d "$DEADLINE_LOCAL" +%s
}

past_deadline() {
  [[ "$(date +%s)" -ge "$(deadline_epoch)" ]]
}

remaining_h() {
  local rem=$(( $(deadline_epoch) - $(date +%s) ))
  printf '%dh%02dm' $((rem / 3600)) $(((rem % 3600) / 60))
}

hours_remaining() {
  echo $(( ($(deadline_epoch) - $(date +%s)) / 3600 ))
}

wait_gpu_free() {
  echo "=== $(date -Is) Attente fin run_048 (poll ${POLL_SEC}s, deadline ${DEADLINE_LOCAL}) ==="
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    if past_deadline; then
      echo "=== $(date -Is) DEADLINE atteinte pendant run_048 — arrêt chaîne ==="
      exit 0
    fi
    pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
    echo "$(date -Is) GPU occupée — ~$(remaining_h) avant libération collègues"
    sleep "$POLL_SEC"
  done
}

wait_gpu_free

if past_deadline; then
  echo "=== $(date -Is) DEADLINE atteinte — rien à lancer ==="
  exit 0
fi

hrs="$(hours_remaining)"
if [[ "$hrs" -ge "$MIN_HOURS_FOR_NEXT" ]]; then
  echo "=== $(date -Is) ~$(remaining_h) restantes — run_050 speechLLM seed2 (≤${MAX_RUN_HOURS} h estimées) ==="
  chmod +x \
    "${ROOT}/scripts/run_modyco_speechllm_14k_seed2.sh" \
    "${ROOT}/2_speechLLM/scripts/run_050_b1_utterance_large_14k_seed2_nohup.sh"
  bash "${ROOT}/scripts/run_modyco_speechllm_14k_seed2.sh"
else
  echo "=== $(date -Is) Moins de ${MIN_HOURS_FOR_NEXT}h avant ${DEADLINE_LOCAL} — run_050 ignoré (GPU libre pour collègues) ==="
fi

echo "=== $(date -Is) CHAÎNE 048→050 TERMINÉE — GPU libre (deadline ${DEADLINE_LOCAL}) ==="
