#!/usr/bin/env bash
# Job OAR IMAG — éval best.pt run_043 ST replicate (après timeout train bf16 OAR 129671).
#
# Soumission :
#   oarsub -S -l /gpu=1,walltime=02:00:00 -p "host='lig-gpu10.imag.fr'" \
#     /home/getalp/bonapelm/S3T/run_oar_eval_043_st_replicate_aker.sh

#OAR -n run_043_eval_st_replicate
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG_CANDIDATES=(
  "1_Transformer/configs/fr-en/base_utterance_large_14k_v5_replicate_aker_bf16.yaml"
  "configs/fr-en/base_utterance_large_14k_v5_replicate_aker_bf16.yaml"
  "1_Transformer/configs/fr-en/base_utterance_large_14k_v5_replicate.yaml"
)
CFG=""
for candidate in "${CFG_CANDIDATES[@]}"; do
  if [[ -f "${candidate}" ]]; then
    CFG="${candidate}"
    break
  fi
done
if [[ -z "${CFG}" ]]; then
  echo "ERROR: config introuvable" >&2
  exit 2
fi

RUN_ID="run_043_aker_st_l14k_v5_replicate"
LOG="logs/run_043_eval_st_replicate.log"
BEST="runs/fr-en/${RUN_ID}/checkpoints/best.pt"

echo "=== $(date -u -Iseconds) EVAL ${RUN_ID} CFG=${CFG} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if [[ ! -f "${BEST}" ]]; then
  echo "ERROR: best.pt absent : ${BEST}" >&2
  exit 2
fi
ls -la "${BEST}"

python 1_Transformer/pipeline.py evaluate \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  --beam-size 5 \
  -v 2>&1 | tee "${LOG}"

echo "=== $(date -u -Iseconds) done ==="
head -1 "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt" || true
head -1 "runs/fr-en/${RUN_ID}/eval/sacrebleu_dev.txt" || true
