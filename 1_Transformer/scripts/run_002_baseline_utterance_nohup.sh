#!/usr/bin/env bash
# Baseline ST Table 8 — utterance (m-TEDx natif) : SPM → train 80k → evaluate beam 5.
#
# Durée estimée : ~8 h GPU (aligné run_001 sentence_like).
#
# Lancement détachable (tour GPU) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 1_Transformer/scripts/run_002_baseline_utterance_nohup.sh \
#     > logs/run_002_transformer_utterance_nohup_wrapper.log 2>&1 &
#   echo $! > logs/run_002_transformer_utterance_nohup.pid

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance.yaml"
RUN="run_002_transformer_baseline_utterance"
MANIFESTS="datasets/manifests/fr-en"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${RUN}_spm_train_eval.log"

TARGET_TXT="${MANIFESTS}/train.target.txt"
if [[ ! -f "$TARGET_TXT" && -f "${MANIFESTS}/train.tsv" ]]; then
  echo "=== Génération ${TARGET_TXT} depuis train.tsv ==="
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

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv — lancer:" >&2
    echo "  python scripts_communs/2_prepare.py --langpair fr-en --segment-mode utterance" >&2
    exit 2
  fi
done

{
  echo "=== $(date -Is) SPM utterance (vocab 1k) ==="
  python 1_Transformer/3_spm.py \
    --langpair fr-en \
    --vocab-size 1000 \
    --manifests-root datasets/manifests \
    --train-text "${TARGET_TXT}" \
    --overwrite

  echo "=== $(date -Is) TRAIN ${RUN} ==="
  python 1_Transformer/pipeline.py train \
    --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate \
    --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
