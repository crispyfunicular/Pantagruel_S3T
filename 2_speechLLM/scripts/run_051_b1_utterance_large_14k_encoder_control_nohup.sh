#!/usr/bin/env bash
# Run 051 — speechLLM B1 utterance L-14k, contrôle encoder_layer -1 (piste J).
# Durée estimée : ~2,5 h GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_encoder_control.yaml"
RUN="run_051_speechllm_b1_utterance_large_14k_encoder_control"
LOG="${ROOT}/logs/${RUN}_train_eval.log"
mkdir -p "${ROOT}/logs"

{
  echo "=== $(date -Is) RUN ${RUN} (L-14k, encoder_layer=-1 contrôle run_012) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
