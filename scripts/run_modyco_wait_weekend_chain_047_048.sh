#!/usr/bin/env bash
# Modyco — chaîne week-end : run_047 (couche 9) → run_048 (couche 6), piste J.
#
# À lancer vendredi soir ou samedi matin quand le GPU est libre (serveur partagé).
# Durée totale estimée : ~12–16 h GPU.
#
# Usage (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_wait_weekend_chain_047_048.sh \
#     > logs/chain_weekend_047_048_modyco_wait.log 2>&1 &
#   tail -f logs/chain_weekend_047_048_modyco_wait.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"

wait_gpu_free() {
  echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s) ==="
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
    echo "$(date -Is) GPU occupée — nouvelle vérif dans ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
}

wait_gpu_free
echo "=== $(date -Is) Lancement run_047 (speechLLM L-14k, encoder_layer=9) ==="
chmod +x \
  "${ROOT}/scripts/run_modyco_speechllm_14k_layer9.sh" \
  "${ROOT}/2_speechLLM/scripts/run_047_b1_utterance_large_14k_layer9_nohup.sh"
bash "${ROOT}/scripts/run_modyco_speechllm_14k_layer9.sh"

wait_gpu_free
echo "=== $(date -Is) Lancement run_048 (speechLLM L-14k, encoder_layer=6) ==="
chmod +x \
  "${ROOT}/scripts/run_modyco_speechllm_14k_layer6.sh" \
  "${ROOT}/2_speechLLM/scripts/run_048_b1_utterance_large_14k_layer6_nohup.sh"
bash "${ROOT}/scripts/run_modyco_speechllm_14k_layer6.sh"

echo "=== $(date -Is) CHAÎNE WEEK-END Modyco (047→048) TERMINÉE ==="
