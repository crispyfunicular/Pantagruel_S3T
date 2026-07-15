#!/usr/bin/env bash
# Job OAR IMAG — baseline ST L-14k v5 (SpecAugment) sur lig-gpu6.
#
# Soumission depuis aker (login shell) :
#   bash -lc 'oarsub -S -l /gpu=1,walltime=24:00:00 -p "host='\''lig-gpu6.imag.fr'\''" ~/S3T/scripts/run_oar_st_l14k_v5_aker.sh'
#
# Run cible : run_060_aker_st_l14k_v5
# Durée estimée : 12–24 h GPU (2080 Ti) ; timeout interne 23 h puis éval best.pt

#OAR -n run_060_st_l14k_v5
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="1_Transformer/configs/fr-en/base_utterance_large_14k_v5_aker.yaml"
RUN_ID="run_060_aker_st_l14k_v5"
LOG="logs/run_060_st_l14k_v5_train_eval.log"
SPM_MODEL="datasets/processed/spm/fr-en_1000.model"
MANIFESTS="datasets/manifests/fr-en"
MAX_HOURS="${MAX_HOURS:-23}"

echo "=== $(date -u -Iseconds) start ST ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for split in train valid test; do
  if [[ ! -f "${MANIFESTS}/${split}.tsv" ]]; then
    echo "ERROR: manquant ${MANIFESTS}/${split}.tsv" >&2
    exit 2
  fi
done

if [[ ! -f "$SPM_MODEL" ]]; then
  TARGET_TXT="${MANIFESTS}/train.target.txt"
  if [[ ! -f "$TARGET_TXT" ]]; then
    python -c "
import csv
from pathlib import Path
manifest = Path('${MANIFESTS}/train.tsv')
target   = Path('${TARGET_TXT}')
with manifest.open(encoding='utf-8') as fi, target.open('w', encoding='utf-8') as fo:
    for row in csv.DictReader(fi, delimiter='\t'):
        fo.write(row['tgt_text'].strip() + '\n')
"
  fi
  python 1_Transformer/3_spm.py \
    --langpair fr-en \
    --vocab-size 1000 \
    --manifests-root datasets/manifests \
    --train-text "$TARGET_TXT" \
    --overwrite
else
  echo "SPM existant : ${SPM_MODEL}"
fi

{
  TRAIN_ARGS=()
  if [[ -f "runs/fr-en/${RUN_ID}/checkpoints/last.pt" ]]; then
    TRAIN_ARGS+=(--resume)
    echo "=== $(date -u -Iseconds) Reprise checkpoint (--resume) ==="
  fi
  timeout "$((MAX_HOURS * 3600))" \
    python 1_Transformer/pipeline.py train \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      "${TRAIN_ARGS[@]}" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -u -Iseconds) TIMEOUT ${MAX_HOURS}h — éval best checkpoint ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      --beam-size 5 -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  else
    echo "=== $(date -u -Iseconds) EVALUATE ${RUN_ID} (beam 5) ==="
    python 1_Transformer/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      --beam-size 5 -v
  fi
  echo "=== $(date -u -Iseconds) done exit=${ec} ==="
} 2>&1 | tee "$LOG"
