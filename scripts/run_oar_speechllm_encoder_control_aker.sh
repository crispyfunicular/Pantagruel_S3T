#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k contrôle encoder_layer -1 (run_078, piste J).
#
# Réplication IMAG de run_051 Modyco (**13,58** test). ~3–4 h GPU (2080 Ti).
#
# Soumission depuis aker :
#   oarsub -l /gpu=1,walltime=8:00:00 -n run_078_speechllm_encoder_control \
#     /home/getalp/bonapelm/S3T/run_oar_speechllm_encoder_control_aker.sh

#OAR -l /gpu=1,walltime=8:00:00
#OAR -n run_078_speechllm_encoder_control
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_encoder_control.yaml"
RUN_ID="run_078_aker_speechllm_l14k_encoder_control"
LOG="logs/run_078_train_eval.log"

echo "=== $(date -u -Iseconds) start encoder control ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
