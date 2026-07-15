#!/usr/bin/env bash
# Job OAR IMAG — relance SeamlessM4T v2 (run_075b, piste K).
#
# Première tentative (2080 Ti) : OOM sur tous les segments → BLEU 0,00.
# Cibler lig-gpu10 (H100) — ~4–8 h GPU.
#
# Soumission depuis aker (login shell) :
#   oarsub -l /gpu=1,walltime=12:00:00 -p "host='lig-gpu10.imag.fr'" \
#     -n run_075b_seamless_aker \
#     /home/getalp/bonapelm/S3T/run_oar_seamless_075b_aker.sh

#OAR -l /gpu=1,walltime=12:00:00
#OAR -p host='lig-gpu10.imag.fr'
#OAR -n run_075b_seamless_aker
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="6_open_baselines/configs/fr-en/seamless_m4t_v2_large.yaml"
RUN_ID="run_075b_aker_open_seamlessm4t_v2_utterance"
LOG="logs/run_075b_seamless_retry.log"

echo "=== $(date -u -Iseconds) start Seamless ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python 6_open_baselines/pipeline.py evaluate \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

path="runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt"
if [[ -f "${path}" ]]; then
  echo "=== ${RUN_ID} : $(head -1 "${path}") ==="
fi

echo "=== $(date -u -Iseconds) done exit=$? ==="
