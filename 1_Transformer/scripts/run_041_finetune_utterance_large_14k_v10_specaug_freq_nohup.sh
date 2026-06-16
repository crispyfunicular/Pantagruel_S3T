#!/usr/bin/env bash
# Run 041 — finetune ST L-14k v10 SpecAugment freq depuis run_026 (~3,5 h GPU).
#
# Reprend run_026/best.pt (@ ~44k updates) et entraîne jusqu'à 69k max (+ SpecAugment fréquentiel).
#
# Lancement nohup (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 1_Transformer/scripts/run_041_finetune_utterance_large_14k_v10_specaug_freq_nohup.sh \
#     > logs/run_041_st_14k_v10_specaug_freq_finetune_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_large_14k_v10_specaug_freq_finetune.yaml"
RUN="run_041_transformer_finetune_utterance_large_14k_v10_specaug_freq_from_run026"
INIT_CKPT="${ROOT}/runs/fr-en/run_026_transformer_baseline_utterance_large_14k_v5/checkpoints/best.pt"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_spm_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_1000.model"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv — prepare utterance d'abord." >&2
    exit 2
  fi
done

if [[ ! -f "$INIT_CKPT" ]]; then
  echo "ERROR: checkpoint run_026 manquant : ${INIT_CKPT}" >&2
  exit 2
fi

{
  echo "=== $(date -Is) SPM existant : ${SPM_MODEL} ==="

  echo "=== $(date -Is) TRAIN ${RUN} (finetune run_026 + SpecAugment freq, max 69k updates) ==="
  python 1_Transformer/pipeline.py train \
    --config "$CFG" \
    --run-id "$RUN" \
    --resume \
    --resume-from "$INIT_CKPT" \
    -v

  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
