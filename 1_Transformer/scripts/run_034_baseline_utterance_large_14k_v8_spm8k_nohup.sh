#!/usr/bin/env bash
# Run 034 — baseline ST utterance L-14k v8 (SPM 8k + SpecAugment, piste 3 Table 8).
#
# Suite run_031 SPM 5k ; vocab cible LeBenchmark 2.0 (~8k sous-mots).
# Durée estimée : ~8–10 h GPU (budget Modyco).
#
# Lancement nohup (Modyco) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 1_Transformer/scripts/run_034_baseline_utterance_large_14k_v8_spm8k_nohup.sh \
#     > logs/run_034_st_14k_v8_wrapper.log 2>&1 &
#   tail -f logs/run_034_transformer_baseline_utterance_large_14k_v8_spm8k_spm_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_large_14k_v8_spm8k.yaml"
RUN="run_034_transformer_baseline_utterance_large_14k_v8_spm8k"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_spm_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_8000.model"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv — prepare utterance d'abord." >&2
    exit 2
  fi
done

{
  if [[ ! -f "$SPM_MODEL" ]]; then
    echo "=== $(date -Is) SPM utterance (vocab 8k) — absent, entraînement SPM ==="
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
      --vocab-size 8000 \
      --manifests-root datasets/manifests \
      --train-text "${TARGET_TXT}" \
      --overwrite
  else
    echo "=== $(date -Is) SPM existant : ${SPM_MODEL} ==="
  fi

  echo "=== $(date -Is) TRAIN ${RUN} (L-14k v8: SPM 8k + SpecAugment) ==="
  python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
