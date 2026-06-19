#!/usr/bin/env bash
# Run 048 — speechLLM B1 utterance L-14k, couche encodeur 6 (piste J).
# Durée estimée : ~6–8 h GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_layer6.yaml"
RUN="run_048_speechllm_b1_utterance_large_14k_layer6"
LOG="${ROOT}/logs/${RUN}_train_eval.log"
mkdir -p "${ROOT}/logs"

{
  echo "=== $(date -Is) RUN ${RUN} (L-14k, encoder_layer=6) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
