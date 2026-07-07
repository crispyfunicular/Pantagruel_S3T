#!/usr/bin/env bash
# OVH — speechLLM L-14k + Mistral-7B 4-bit (run_062 seul).
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_speechllm_062_mistral_l14k.sh \
#     > logs/run_062_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_MISTRAL="${MAX_HOURS_MISTRAL:-15}"

if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
  echo "ERROR: GPU occupé :" >&2
  pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
  exit 2
fi

CFG_SLM="2_speechLLM/configs/fr-en/b1_utterance_large_14k_mistral_7b_ovh.yaml"
RUN_SLM="run_062_speechllm_b2bis_utterance_large_14k_mistral_7b_ovh"
LOG_SLM="logs/${RUN_SLM}_train_eval.log"

echo "=== $(date -Is) Dry-run ${RUN_SLM} ==="
python 2_speechLLM/pipeline.py train --config "$CFG_SLM" --run-id "$RUN_SLM" --dry-run

echo "=== $(date -Is) LANCEMENT ${RUN_SLM} (max ${MAX_HOURS_MISTRAL}h) ==="
{
  timeout "$((MAX_HOURS_MISTRAL * 3600))" \
    python 2_speechLLM/pipeline.py run \
      --config "$CFG_SLM" \
      --run-id "$RUN_SLM" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT ${MAX_HOURS_MISTRAL}h — éval best checkpoint ==="
    python 2_speechLLM/pipeline.py evaluate \
      --config "$CFG_SLM" \
      --run-id "$RUN_SLM" -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  fi
} 2>&1 | tee -a "$LOG_SLM"

BLEU_SLM="n/a"
if [[ -f "runs/fr-en/${RUN_SLM}/eval/sacrebleu_test.txt" ]]; then
  BLEU_SLM="$(head -1 "runs/fr-en/${RUN_SLM}/eval/sacrebleu_test.txt")"
fi
echo "=== $(date -Is) ${RUN_SLM} — BLEU test : ${BLEU_SLM} ==="
