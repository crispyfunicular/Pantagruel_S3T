#!/usr/bin/env bash
# Modyco — chaîne nocturne : run_006 (en cours) puis run_049 ST seed 2.
#
# Deadline GPU : DEADLINE_LOCAL (défaut 2026-06-19 10:00:00) — libération pour collègues.
#
# Usage (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_wait_chain_night_until_10h.sh \
#     > logs/chain_night_until_10h_modyco_wait.log 2>&1 &
#   tail -f logs/chain_night_until_10h_modyco_wait.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
DEADLINE_LOCAL="${DEADLINE_LOCAL:-2026-06-19 10:00:00}"
MIN_HOURS_FOR_SEED2="${MIN_HOURS_FOR_SEED2:-6}"

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
  echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s, deadline ${DEADLINE_LOCAL}) ==="
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    if past_deadline; then
      echo "=== $(date -Is) DEADLINE atteinte — arrêt chaîne ==="
      exit 0
    fi
    pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
    echo "$(date -Is) GPU occupée — ~$(remaining_h) restantes"
    sleep "$POLL_SEC"
  done
}

wait_gpu_free

if past_deadline; then
  echo "=== $(date -Is) DEADLINE atteinte — rien à lancer ==="
  exit 0
fi

if [[ "$(hours_remaining)" -ge "$MIN_HOURS_FOR_SEED2" ]]; then
  echo "=== $(date -Is) ~$(remaining_h) restantes — run_049 ST v5 seed2 (~7–8 h) ==="
  chmod +x "${ROOT}/scripts/run_modyco_st_14k_v5_seed2.sh"
  bash "${ROOT}/scripts/run_modyco_st_14k_v5_seed2.sh"
else
  echo "=== $(date -Is) Moins de ${MIN_HOURS_FOR_SEED2}h avant deadline — run_049 ignoré ==="
fi

echo "=== $(date -Is) CHAÎNE NOCTURNE TERMINÉE — GPU libre (deadline ${DEADLINE_LOCAL}) ==="
