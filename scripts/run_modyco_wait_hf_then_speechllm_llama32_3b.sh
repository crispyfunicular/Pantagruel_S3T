#!/usr/bin/env bash
# Modyco — attend l'accès HF Llama-3.2-3B puis lance run_052 (max 14 h GPU).
#
# Utile quand la licence est acceptée mais l'approbation Meta est encore « pending ».
#
# Usage (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_wait_hf_then_speechllm_llama32_3b.sh \
#     > logs/run_052_modyco_wait_hf_chain.log 2>&1 &
#   tail -f logs/run_052_modyco_wait_hf_chain.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
MAX_WAIT_HOURS="${MAX_WAIT_HOURS:-48}"

llama_access_ok() {
  python - <<'PY' >/dev/null 2>&1
from huggingface_hub import hf_hub_download

hf_hub_download("meta-llama/Llama-3.2-3B-Instruct", "config.json")
PY
}

wait_llama_access() {
  local deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
  echo "=== $(date -Is) Attente accès HF meta-llama/Llama-3.2-3B-Instruct (poll ${POLL_SEC}s, max ${MAX_WAIT_HOURS}h) ==="
  while ! llama_access_ok; do
    if [[ "$(date +%s)" -ge "$deadline" ]]; then
      echo "=== $(date -Is) TIMEOUT attente approbation Meta (${MAX_WAIT_HOURS}h) ===" >&2
      exit 3
    fi
    echo "$(date -Is) Accès Llama pas encore actif (approbation Meta en cours ?) — attente ${POLL_SEC}s"
    sleep "$POLL_SEC"
  done
  echo "=== $(date -Is) Accès Llama OK — lancement run_052 ==="
}

wait_llama_access
chmod +x "${ROOT}/scripts/run_modyco_speechllm_14k_llama32_3b.sh"
exec bash "${ROOT}/scripts/run_modyco_speechllm_14k_llama32_3b.sh"
