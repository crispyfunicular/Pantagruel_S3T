#!/usr/bin/env bash
# Modyco — Piste D (P0) : réévaluation beam 5 avec last.pt vs best.pt.
#
# Durée attendue : ~30–60 min GPU (pas d'entraînement).
# Référence : documentation/recommandations.md § Piste D.
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate
#   nohup bash scripts/run_modyco_eval_piste_d_lastpt.sh \
#     > logs/run_piste_d_eval_lastpt_wrapper.log 2>&1 &
#   tail -f logs/run_piste_d_eval_lastpt.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run_piste_d_eval_lastpt.log"

eval_lastpt() {
  local cfg="$1"
  local src_run="$2"
  local eval_run="$3"
  local ckpt="${ROOT}/runs/fr-en/${src_run}/checkpoints/last.pt"
  local out="${ROOT}/runs/fr-en/${eval_run}/eval/sacrebleu_test.txt"

  if [[ ! -f "$ckpt" ]]; then
    echo "SKIP ${src_run}: checkpoint manquant (${ckpt})" | tee -a "$LOG"
    return 0
  fi
  if [[ -f "$out" ]]; then
    echo "=== $(date -Is) ${eval_run} déjà évalué ===" | tee -a "$LOG"
    head -1 "$out" | tee -a "$LOG"
    return 0
  fi

  echo "=== $(date -Is) EVALUATE ${eval_run} (beam 5, last.pt de ${src_run}) ===" | tee -a "$LOG"
  python 1_Transformer/pipeline.py evaluate \
    --config "$cfg" \
    --run-id "$eval_run" \
    --checkpoint "$ckpt" \
    --beam-size 5 \
    -v 2>&1 | tee -a "$LOG"
}

{
  echo "=== $(date -Is) Piste D — rééval last.pt (greedy best vs beam test) ==="

  eval_lastpt \
    "1_Transformer/configs/fr-en/base_utterance_large_14k_v5.yaml" \
    "run_026_transformer_baseline_utterance_large_14k_v5" \
    "run_026_eval_lastpt"

  eval_lastpt \
    "1_Transformer/configs/fr-en/base_utterance_large_14k_v9_specaug_strong.yaml" \
    "run_037_transformer_baseline_utterance_large_14k_v9_specaug_strong" \
    "run_037_eval_lastpt"

  echo "=== $(date -Is) DONE Piste D ==="
} 2>&1 | tee -a "$LOG"
