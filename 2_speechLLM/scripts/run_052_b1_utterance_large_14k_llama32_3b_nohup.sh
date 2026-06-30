#!/usr/bin/env bash
# Run 052 — speechLLM B2bis utterance L-14k + Llama-3.2-3B-Instruct.
#
# Référence : run_012 Phi-2 @ 15,03 test ; run_018 Qwen2.5-3B @ 12,95 test.
# Durée estimée : ~5–10 h GPU (encodeur L-14k + LLM 3B, batch 1).
# Prérequis : licence Llama 3.2 acceptée + HF_TOKEN ou huggingface-cli login.
#
# Lancement détachable (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_052_b1_utterance_large_14k_llama32_3b_nohup.sh \
#     > logs/run_052_speechllm_llama_wrapper.log 2>&1 &
#   echo $! > logs/run_052_speechllm_llama_nohup.pid
#   tail -f logs/run_052_speechllm_b2bis_utterance_large_14k_llama32_3b_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_llama32_3b.yaml"
RUN="run_052_speechllm_b2bis_utterance_large_14k_llama32_3b"
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
  echo "=== $(date -Is) RUN ${RUN} (L-14k gelé + Llama-3.2-3B gelé, utterance, max ${MAX_HOURS}h) ==="
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
