#!/usr/bin/env bash
# Run 050 — speechLLM B1 utterance L-14k, 2e seed (réplicabilité run_012).
# Durée estimée : ~2,5–6 h GPU (max ~8 h).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_seed2.yaml"
RUN="run_050_speechllm_b1_utterance_large_14k_seed2"
LOG="${ROOT}/logs/${RUN}_train_eval.log"
mkdir -p "${ROOT}/logs"

{
  echo "=== $(date -Is) RUN ${RUN} (L-14k gelé, seed=1 vs run_012 seed=42) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
