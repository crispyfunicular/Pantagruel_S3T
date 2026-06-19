#!/usr/bin/env bash
# Run 006 — speechLLM B1 utterance B-1k, encodeur dégelé (LR 5e-5, WD 0.01).
#
# Durée estimée : ~3–4 h GPU.
#
# Lancement nohup (Modyco) :
#   nohup bash 2_speechLLM/scripts/run_006_b1_utterance_unfreeze_nohup.sh \
#     > logs/run_006_speechllm_unfreeze_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_unfreeze.yaml"
RUN="run_006_speechllm_b1_utterance_unfreeze"
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
  echo "=== $(date -Is) RUN ${RUN} (B-1k unfreeze utterance, LR 5e-5) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
