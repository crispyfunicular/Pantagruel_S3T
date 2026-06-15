#!/usr/bin/env bash
# Run 023 — speechLLM B1 utterance L-14k (réplication run_012, max 48 tok).
#
# Budget Modyco : ≤ 3 h GPU. Durée estimée ~1,5–2 h (aligné run_012 OVH).
#
# Lancement nohup (Modyco) :
#   nohup bash 2_speechLLM/scripts/run_023_b1_utterance_large_14k_replicate_nohup.sh \
#     > logs/run_023_speechllm_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_replicate.yaml"
RUN="run_023_speechllm_b1_utterance_large_14k_replicate"
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
  echo "=== $(date -Is) RUN ${RUN} (L-14k gelé, max 48 tok — réplication run_012) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
