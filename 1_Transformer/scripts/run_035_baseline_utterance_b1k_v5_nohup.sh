#!/usr/bin/env bash
# Run 035 — baseline ST utterance B-1k v5 (SpecAugment, budget ≤ 3,5 h GPU).
#
# Encodeur Base (~1k h pré-train) ; cible papier ~17,5 BLEU test.
# Durée estimée : ~2–3 h (run_004 v2 sans SpecAugment : ~1 h 15).
#
# Lancement nohup (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 1_Transformer/scripts/run_035_baseline_utterance_b1k_v5_nohup.sh \
#     > logs/run_035_st_b1k_v5_wrapper.log 2>&1 &
#   tail -f logs/run_035_transformer_baseline_utterance_b1k_v5_spm_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_b1k_v5.yaml"
RUN="run_035_transformer_baseline_utterance_b1k_v5"
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

  echo "=== $(date -Is) TRAIN ${RUN} (B-1k v5: SpecAugment) ==="
  python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
