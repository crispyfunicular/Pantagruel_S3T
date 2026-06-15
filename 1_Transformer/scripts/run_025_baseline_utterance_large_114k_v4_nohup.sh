#!/usr/bin/env bash
# Run 025 — baseline ST utterance L-114k v4 (batch effectif 64, piste 1 Table 8).
#
# Changements vs v3 (run_019) : grad_accum 64, LR 2e-4, warmup 1k, 10k updates
# (même volume de micro-batches que run_019 : 80k×8 = 10k×64).
# Durée estimée : ~9–10 h GPU (OVH).
#
# Lancement nohup (OVH) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_ovh_st_114k_v4.sh > logs/run_025_ovh_chain_wrapper.log 2>&1 &
#   tail -f logs/run_025_transformer_baseline_utterance_large_114k_v4_spm_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_large_114k_v4.yaml"
RUN="run_025_transformer_baseline_utterance_large_114k_v4"
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

  echo "=== $(date -Is) TRAIN ${RUN} (L-114k v4: batch eff. 64, LR 2e-4) ==="
  python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) EVALUATE ${RUN} (beam 5) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
