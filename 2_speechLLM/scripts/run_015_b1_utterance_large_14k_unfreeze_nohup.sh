#!/usr/bin/env bash
# Run 015 — speechLLM B1 utterance L-14k, encodeur dégelé + décodage 128 tokens.
#
# Pistes variantes.md §2 : dégel encodeur utterance, max_new_tokens ↑, early stop.
# Durée estimée : ~2–3,5 h GPU (budget utilisateur ≤ 4 h).
#
# Lancement détachable (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_015_b1_utterance_large_14k_unfreeze_nohup.sh \
#     > logs/run_015_speechllm_unfreeze_wrapper.log 2>&1 &
#   tail -f logs/run_015_speechllm_b1_utterance_large_14k_unfreeze_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_unfreeze.yaml"
RUN="run_015_speechllm_b1_utterance_large_14k_unfreeze"
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
  echo "=== $(date -Is) RUN ${RUN} (L-14k unfreeze, max_new_tokens=128, early stop=3) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
