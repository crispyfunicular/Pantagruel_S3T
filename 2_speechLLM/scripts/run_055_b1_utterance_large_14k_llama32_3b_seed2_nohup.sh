#!/usr/bin/env bash
# Run 055 — speechLLM B2bis utterance L-14k + Llama-3.2-3B-Instruct (seed 2).
#
# Réplicabilité run_052 (seed 42 @ **16,31** test) ; seul changement : seed 1.
# Durée estimée : ~3 h GPU (early stop typique ~10k updates).
# Prérequis : licence Llama 3.2 + HF_TOKEN ou huggingface-cli login.
#
# Lancement nohup (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_055_b1_utterance_large_14k_llama32_3b_seed2_nohup.sh \
#     > logs/run_055_speechllm_llama_seed2_wrapper.log 2>&1 &
#   tail -f logs/run_055_speechllm_b2bis_utterance_large_14k_llama32_3b_seed2_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_llama32_3b_seed2.yaml"
RUN="run_055_speechllm_b2bis_utterance_large_14k_llama32_3b_seed2"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_train_eval.log"
MAX_HOURS="${MAX_RUN_HOURS:-14}"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv — prepare utterance d'abord." >&2
    echo "  python scripts_communs/pipeline.py prepare --langpair fr-en" >&2
    exit 2
  fi
done

{
  echo "=== $(date -Is) RUN ${RUN} (L-14k gelé + Llama-3.2-3B gelé, seed 1, utterance, max ${MAX_HOURS}h) ==="
  timeout "$((MAX_HOURS * 3600))" \
    python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT après ${MAX_HOURS}h ===" >&2
    exit 124
  fi
  if [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  fi
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
