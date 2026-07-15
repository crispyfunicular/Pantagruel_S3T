#!/usr/bin/env bash
# OVH — ST L-14k v13 batch 32 safe (run_065, piste B relance).
#
# Base run_046 (collapse 2,76) → warmup 15k + LR 5e-5. ~8–14 h GPU V100.
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_ovh_st_065_l14k_batch32_safe.sh \
#     > logs/run_065_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_ST="${MAX_HOURS_ST:-16}"

CFG="1_Transformer/configs/fr-en/base_utterance_large_14k_v13_batch32_safe.yaml"
RUN_ID="run_065_transformer_baseline_utterance_large_14k_v13_batch32_safe"
LOG="logs/${RUN_ID}_train_eval.log"

if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
  echo "ERROR: GPU occupé :" >&2
  pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
  exit 2
fi

echo "=== $(date -Is) Dry-run ${RUN_ID} ==="
python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN_ID" --dry-run

echo "=== $(date -Is) LANCEMENT ${RUN_ID} (max ${MAX_HOURS_ST}h) ==="
{
  timeout "$((MAX_HOURS_ST * 3600))" \
    python 1_Transformer/pipeline.py train \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT ${MAX_HOURS_ST}h — éval best checkpoint ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      --beam-size 5 -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  else
    echo "=== $(date -Is) EVALUATE ${RUN_ID} (beam 5) ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      --beam-size 5 -v
  fi
} 2>&1 | tee -a "$LOG"

BLEU="n/a"
[[ -f "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt" ]] \
  && BLEU="$(head -1 "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt")"
echo "=== $(date -Is) ${RUN_ID} — BLEU test : ${BLEU} ==="
