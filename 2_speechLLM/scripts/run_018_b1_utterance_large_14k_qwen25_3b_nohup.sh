#!/usr/bin/env bash
# Run 018 — speechLLM B2bis utterance L-14k + Qwen2.5-3B-Instruct (Phi-2 ablation).
#
# Référence : run_012 Phi-2 gelé @ 15,03 test (OVH) ; run_014/015 v2/unfreeze échoués (Modyco).
# Durée estimée : ~5–8 h GPU (encodeur L-14k + LLM 3B, batch 1).
#
# Lancement détachable (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 2_speechLLM/scripts/run_018_b1_utterance_large_14k_qwen25_3b_nohup.sh \
#     > logs/run_018_speechllm_qwen_wrapper.log 2>&1 &
#   echo $! > logs/run_018_speechllm_qwen_nohup.pid
#   tail -f logs/run_018_speechllm_b2bis_utterance_large_14k_qwen25_3b_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_qwen25_3b.yaml"
RUN="run_018_speechllm_b2bis_utterance_large_14k_qwen25_3b"
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
  echo "=== $(date -Is) RUN ${RUN} (L-14k gelé + Qwen2.5-3B gelé, utterance) ==="
  python 2_speechLLM/pipeline.py run --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
