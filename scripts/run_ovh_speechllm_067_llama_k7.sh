#!/usr/bin/env bash
# OVH — speechLLM L-14k + Llama-3.2-3B, downsample k=7, 128 tokens (run_067, piste L).
#
# Base run_052 (k=5, 48 tok). ~3–6 h GPU. OOM sur IMAG 2080 Ti.
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_speechllm_067_llama_k7.sh \
#     > logs/run_067_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_SLM="${MAX_HOURS_SLM:-10}"

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_llama32_3b_k7.yaml"
RUN_ID="run_067_speechllm_b1_utterance_large_14k_llama32_3b_k7"
LOG="logs/${RUN_ID}_train_eval.log"

if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
  echo "ERROR: GPU occupé :" >&2
  pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
  exit 2
fi

if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "WARN: HF_TOKEN absent — Llama-3.2-3B est gated." >&2
fi

echo "=== $(date -Is) Dry-run ${RUN_ID} ==="
python 2_speechLLM/pipeline.py train --config "$CFG" --run-id "$RUN_ID" --dry-run

echo "=== $(date -Is) LANCEMENT ${RUN_ID} (max ${MAX_HOURS_SLM}h) ==="
{
  timeout "$((MAX_HOURS_SLM * 3600))" \
    python 2_speechLLM/pipeline.py run \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT ${MAX_HOURS_SLM}h — éval best checkpoint ==="
    python 2_speechLLM/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  fi
} 2>&1 | tee -a "$LOG"

BLEU="n/a"
[[ -f "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt" ]] \
  && BLEU="$(head -1 "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt")"
echo "=== $(date -Is) ${RUN_ID} — BLEU test : ${BLEU} ==="
