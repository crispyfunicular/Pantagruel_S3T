#!/usr/bin/env bash
# Run 056 — speechLLM B1 utterance L-114k, couche encodeur 9 (piste J, OVH).
#
# Référence : run_013 (**15,24** test, couche -1) ; run_047 L-14k couche 9 (**14,00**).
# Durée estimée : ~3–5 h GPU.
#
# Lancement nohup (OVH) :
#   nohup bash 2_speechLLM/scripts/run_056_b1_utterance_large_114k_layer9_nohup.sh \
#     > logs/run_056_speechllm_layer9_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_114k_layer9.yaml"
RUN="run_056_speechllm_b1_utterance_large_114k_layer9"
LOG="${ROOT}/logs/${RUN}_train_eval.log"
mkdir -p "${ROOT}/logs"

{
  echo "=== $(date -Is) RUN ${RUN} (L-114k, encoder_layer=9) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
