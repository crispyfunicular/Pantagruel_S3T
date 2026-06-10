#!/usr/bin/env bash
# Run 014 — speechLLM B1 utterance L-14k v2 (retry après run_012).
#
# Améliorations vs run_012 :
#   - max_new_tokens : 128 (vs 48)
#   - early_stopping_patience : 3 (arrêt si BLEU dev stagne 3 évals)
#
# Durée estimée : ~1,5–3 h GPU (run_012 ≈ 1,4 h avec max 48 tok ; budget utilisateur ≤ 12 h).
#
# Lancement détachable :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_014_b1_utterance_large_14k_v2_nohup.sh \
#     > logs/run_014_speechllm_14k_v2_wrapper.log 2>&1 &
#   echo $! > logs/run_014_speechllm_14k_v2_nohup.pid
#   tail -f logs/run_014_speechllm_b1_utterance_large_14k_v2_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_v2.yaml"
RUN="run_014_speechllm_b1_utterance_large_14k_v2"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_train_eval.log"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv — prepare utterance d'abord." >&2
    echo "  python scripts_communs/pipeline.py prepare --langpair fr-en" >&2
    exit 2
  fi
done

{
  echo "=== $(date -Is) RUN ${RUN} (L-14k v2: max_new_tokens=128, early stop patience=3) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
