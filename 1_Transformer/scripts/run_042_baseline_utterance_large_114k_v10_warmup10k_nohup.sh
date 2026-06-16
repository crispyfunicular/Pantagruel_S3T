#!/usr/bin/env bash
# Run 042 — baseline ST utterance L-114k v10 warmup 10k + SpecAugment.
#
# Hypothèse : L-114k a besoin d'un warmup plus long que L-14k pour stabiliser
# le fine-tuning (run_028 v5 @ 23,51 < run_026 L-14k @ 26,12).
# Durée estimée : ~10–12 h GPU (OVH).
#
# Lancement nohup (OVH) :
#   nohup bash 1_Transformer/scripts/run_042_baseline_utterance_large_114k_v10_warmup10k_nohup.sh \
#     > logs/run_042_st_114k_v10_warmup10k_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_large_114k_v10_warmup10k.yaml"
RUN="run_042_transformer_baseline_utterance_large_114k_v10_warmup10k"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_spm_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_1000.model"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv" >&2
    exit 2
  fi
done

{
  echo "=== $(date -Is) SPM existant : ${SPM_MODEL} ==="

  echo "=== $(date -Is) TRAIN ${RUN} (L-114k v10: warmup 10k + SpecAugment) ==="
  python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
