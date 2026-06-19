#!/usr/bin/env bash
# Modyco — après run_046 : run_006 speechLLM B-1k dégel utterance.
#
# Contrainte : run_046 + run_006 doivent tenir dans MAX_HOURS_GPU (défaut 11 h)
# depuis le début de run_046, pour libérer le GPU aux collègues.
#
# Usage (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_wait_after_046_then_006.sh \
#     > logs/chain_046_006_modyco_wait.log 2>&1 &
#   tail -f logs/chain_046_006_modyco_wait.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
MAX_HOURS_GPU="${MAX_HOURS_GPU:-11}"
RUN_046_LOG="${ROOT}/logs/run_046_transformer_baseline_utterance_large_14k_v11_batch32_spm_train_eval.log"
CHAIN_START="${CHAIN_START:-}"

if [[ -z "$CHAIN_START" && -f "$RUN_046_LOG" ]]; then
  CHAIN_START="$(stat -c '%Y' "$RUN_046_LOG" 2>/dev/null || date +%s)"
else
  CHAIN_START="${CHAIN_START:-$(date +%s)}"
fi

deadline_epoch() {
  echo $((CHAIN_START + MAX_HOURS_GPU * 3600))
}

past_deadline() {
  [[ "$(date +%s)" -ge "$(deadline_epoch)" ]]
}

wait_gpu_free() {
  echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s) ==="
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    if past_deadline; then
      echo "=== $(date -Is) DEADLINE ${MAX_HOURS_GPU}h atteinte — abandon run_006 ==="
      exit 0
    fi
    pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
    echo "$(date -Is) GPU occupée — nouvelle vérif dans ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

remaining_h() {
  local rem=$(( $(deadline_epoch) - $(date +%s) ))
  printf '%dh%02dm' $((rem / 3600)) $(((rem % 3600) / 60))
}

wait_gpu_free

if past_deadline; then
  echo "=== $(date -Is) DEADLINE ${MAX_HOURS_GPU}h atteinte — run_006 non lancé ==="
  exit 0
fi

echo "=== $(date -Is) GPU libre — ~$(remaining_h) h restantes avant deadline — run_006 (B-1k dégel, ~3–4 h) ==="
chmod +x "${ROOT}/scripts/run_modyco_speechllm_b1_utterance_unfreeze.sh"
bash "${ROOT}/scripts/run_modyco_speechllm_b1_utterance_unfreeze.sh"

echo "=== $(date -Is) CHAÎNE Modyco run_046→run_006 TERMINÉE — GPU libre pour collègues ==="
