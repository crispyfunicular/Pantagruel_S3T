#!/usr/bin/env bash
# Run 036 — finir l'entraînement ST L-14k v9 warmup 10k (Modyco).
#
# Usage :
#   Reprise depuis last.pt / best.pt (recommandé si interruption récente) :
#     nohup bash 1_Transformer/scripts/run_036_baseline_utterance_large_14k_v9_warmup10k_finish_nohup.sh \
#       > logs/run_036_st_14k_v9_warmup10k_finish_wrapper.log 2>&1 &
#
#   Repartir de zéro (--overwrite, supprime les checkpoints existants) :
#     RESUME=0 OVERWRITE=1 nohup bash ... &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_large_14k_v9_warmup10k.yaml"
RUN="run_036_transformer_baseline_utterance_large_14k_v9_warmup10k"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_spm_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_1000.model"

RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"

TRAIN_FLAGS=(--config "$CFG" --run-id "$RUN" -v)
if [[ "$RESUME" == "1" ]]; then
  TRAIN_FLAGS+=(--resume)
elif [[ "$OVERWRITE" == "1" ]]; then
  TRAIN_FLAGS+=(--overwrite)
fi

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv — prepare utterance d'abord." >&2
    exit 2
  fi
done

{
  if [[ ! -f "$SPM_MODEL" ]]; then
    echo "=== $(date -Is) SPM utterance (vocab 1k) — absent, entraînement SPM ==="
    TARGET_TXT="${MANIFESTS}/train.target.txt"
    if [[ ! -f "$TARGET_TXT" && -f "${MANIFESTS}/train.tsv" ]]; then
      python -c "
import csv
from pathlib import Path
manifest = Path('${MANIFESTS}/train.tsv')
target = Path('${TARGET_TXT}')
with manifest.open(encoding='utf-8') as handle_in, target.open('w', encoding='utf-8') as handle_out:
    for row in csv.DictReader(handle_in, delimiter='\t'):
        handle_out.write(row['tgt_text'].strip() + '\n')
"
    fi
    python 1_Transformer/3_spm.py \
      --langpair fr-en \
      --vocab-size 1000 \
      --manifests-root datasets/manifests \
      --train-text "${TARGET_TXT}" \
      --overwrite
  else
    echo "=== $(date -Is) SPM existant : ${SPM_MODEL} ==="
  fi

  echo "=== $(date -Is) TRAIN ${RUN} (flags: resume=${RESUME}, overwrite=${OVERWRITE}) ==="
  python 1_Transformer/pipeline.py train "${TRAIN_FLAGS[@]}"

  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
