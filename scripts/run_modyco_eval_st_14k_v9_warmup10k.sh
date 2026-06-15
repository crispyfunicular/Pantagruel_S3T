#!/usr/bin/env bash
# Modyco — évaluation beam 5 de run_036 (interrompu, best.pt sauvegardé).
#
# Usage :
#   cd ~/S3T && source .venv/bin/activate
#   bash scripts/run_modyco_eval_st_14k_v9_warmup10k.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_large_14k_v9_warmup10k.yaml"
RUN="run_036_transformer_baseline_utterance_large_14k_v9_warmup10k"
CKPT="${ROOT}/runs/fr-en/${RUN}/checkpoints/best.pt"

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint manquant : ${CKPT}" >&2
  exit 2
fi

if [[ -f "${ROOT}/runs/fr-en/${RUN}/eval/sacrebleu_test.txt" ]]; then
  echo "=== $(date -Is) run_036 déjà évalué ==="
  head -1 "${ROOT}/runs/fr-en/${RUN}/eval/sacrebleu_test.txt"
  exit 0
fi

echo "=== $(date -Is) EVALUATE ${RUN} (beam 5, checkpoint best.pt) ==="
python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

echo "=== $(date -Is) DONE ==="
