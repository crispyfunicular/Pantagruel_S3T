#!/usr/bin/env bash
# Run 039 — speechLLM B1 utterance L-14k + SpecAugment (run_023 + masquage temporel).
#
# Budget Modyco : ≤ 3 h GPU. Durée estimée ~2 h.
#
# Lancement nohup (Modyco) :
#   nohup bash 2_speechLLM/scripts/run_039_b1_utterance_large_14k_v5_specaug_nohup.sh \
#     > logs/run_039_speechllm_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_v5_specaug.yaml"
RUN="run_039_speechllm_b1_utterance_large_14k_v5_specaug"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_train_eval.log"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv" >&2
    exit 2
  fi
done

{
  echo "=== $(date -Is) RUN ${RUN} (L-14k gelé, SpecAugment, max 48 tok) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
