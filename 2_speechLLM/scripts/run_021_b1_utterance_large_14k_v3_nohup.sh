#!/usr/bin/env bash
# Run 021 — speechLLM B1 utterance L-14k v3 (Phi-2 gelé, eval dev complet).
#
# Retry après run_014 v2 (4,09 test) avec max_eval_batches null (leçon run_020 ST).
# Durée estimée : ~2–4 h GPU (run_012 gelé ≈ 1,4 h avec max 48 tok).
#
# Lancement nohup (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_021_b1_utterance_large_14k_v3_nohup.sh \
#     > logs/run_021_speechllm_14k_v3_wrapper.log 2>&1 &
#   tail -f logs/run_021_speechllm_b1_utterance_large_14k_v3_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_v3.yaml"
RUN="run_021_speechllm_b1_utterance_large_14k_v3"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_train_eval.log"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv — prepare utterance d'abord." >&2
    exit 2
  fi
done

{
  echo "=== $(date -Is) RUN ${RUN} (L-14k v3: max 128 tok, eval dev complet) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
