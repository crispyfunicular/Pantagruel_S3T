#!/usr/bin/env bash
# Run 012 — speechLLM B1 utterance, encodeur Pantagruel-L-14k (speech-large-14K).
# Protocole Table 8 : train 20k updates puis evaluate (SacreBLEU, beam 1).
#
# Prérequis : manifests utterance fr-en (scripts_communs/pipeline.py prepare --langpair fr-en).
# Durée estimée : plusieurs heures GPU (encodeur Large, batch 1, grad_acc 8).
#
# Lancement détachable (fermer le laptop) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_012_speechllm_b1_utterance_large_14k_nohup.sh \
#     > logs/run_012_wrapper.log 2>&1 &
#   echo $! > logs/run_012_nohup.pid
#   tail -f logs/run_012_speechllm_b1_utterance_large_14k_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k.yaml"
RUN="run_012_speechllm_b1_utterance_large_14k"
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
  echo "=== $(date -Is) RUN ${RUN} (Pantagruel-L-14k, train → evaluate) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
