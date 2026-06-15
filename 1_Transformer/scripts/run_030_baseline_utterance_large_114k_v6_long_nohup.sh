#!/usr/bin/env bash
# Run 030 — baseline ST utterance L-114k v6 long (PRD 120k + SpecAugment).
#
# Durée estimée : ~15–20 h GPU (OVH).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_large_114k_v6_long.yaml"
RUN="run_030_transformer_baseline_utterance_large_114k_v6_long"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_spm_train_eval.log"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv" >&2
    exit 2
  fi
done

{
  echo "=== $(date -Is) TRAIN ${RUN} (L-114k v6 long: 120k upd, SpecAugment) ==="
  python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN" -v
  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v
  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
