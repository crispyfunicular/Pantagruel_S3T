#!/usr/bin/env bash
# Baseline ST Table 8 — utterance, Pantagruel-L-114k v2 (correctifs anti-collapse).
#
# Pistes variantes.md §1 : encodeur 114k + affinage entraînement (gel 5k, LR 1e-4, early stop).
# Successeur logique de run_014 L-14k v2 (17,21 test) vers cible papier ~25 BLEU.
#
# Durée estimée : ~4–12 h GPU (early stop possible avant 80k).
#
# Lancement nohup (OVH) :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash 1_Transformer/scripts/run_016_baseline_utterance_large_114k_v2_nohup.sh \
#     > logs/run_016_st_114k_v2_wrapper.log 2>&1 &
#   tail -f logs/run_016_transformer_baseline_utterance_large_114k_v2_spm_train_eval.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

CFG="1_Transformer/configs/fr-en/base_utterance_large_114k_v2.yaml"
RUN="run_016_transformer_baseline_utterance_large_114k_v2"
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

  echo "=== $(date -Is) TRAIN ${RUN} (L-114k v2: freeze 5k, early stop, LR 1e-4) ==="
  python 1_Transformer/pipeline.py train --config "$CFG" --run-id "$RUN" -v

  echo "=== $(date -Is) EVALUATE ${RUN} (greedy, beam 5 journalisé) ==="
  python 1_Transformer/pipeline.py evaluate --config "$CFG" --run-id "$RUN" --beam-size 5 -v

  echo "=== $(date -Is) DONE ==="
} 2>&1 | tee -a "$LOG"
