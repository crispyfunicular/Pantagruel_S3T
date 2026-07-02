#!/usr/bin/env bash
# Run 053 — speechLLM B2bis utterance L-114k + Llama-3.2-3B-Instruct.
#
# Suite de run_052 (L-14k + Llama @ 16,31 test) : seul l'encodeur passe en L-114k.
# Référence Phi-2 : run_012 L-14k 15,03 → run_013 L-114k 15,24.
#
# Durée estimée : ~3–5 h GPU (encodeur L-114k + LLM 3B, batch 1).
# Cible : **OVH** (HF `speech-large-114K` gated sur Modyco).
# Prérequis : HF_TOKEN + licence Llama 3.2 acceptée.
#
# Lancement nohup (OVH) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_053_b1_utterance_large_114k_llama32_3b_nohup.sh \
#     > logs/run_053_speechllm_llama_wrapper.log 2>&1 &
#   tail -f logs/run_053_speechllm_b2bis_utterance_large_114k_llama32_3b_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_114k_llama32_3b.yaml"
RUN="run_053_speechllm_b2bis_utterance_large_114k_llama32_3b"
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
  echo "=== $(date -Is) RUN ${RUN} (L-114k gelé + Llama-3.2-3B gelé, utterance, max ${MAX_HOURS}h) ==="
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
