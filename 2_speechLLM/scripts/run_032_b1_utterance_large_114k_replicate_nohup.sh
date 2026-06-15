#!/usr/bin/env bash
# Run 032 — speechLLM B1 utterance L-114k (réplication run_013, max 48 tok).
#
# Budget OVH : ~4–6 h GPU. Objectif : ~15 BLEU test (run_013 @ 15,24).
# run_022 v3 (128 tok) a échoué @ 4,78 test.
#
# Lancement nohup (OVH) :
#   nohup bash 2_speechLLM/scripts/run_032_b1_utterance_large_114k_replicate_nohup.sh \
#     > logs/run_032_speechllm_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_114k_replicate.yaml"
RUN="run_032_speechllm_b1_utterance_large_114k_replicate"
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
  echo "=== $(date -Is) RUN ${RUN} (L-114k gelé, max 48 tok — réplication run_013) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
