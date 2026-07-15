#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k + Phi-2 seed 42, 4e réplication IMAG (run_079, piste F).
#
# Références : run_059 **14,77** ; run_068 **13,71** ; run_071 **14,69**.
#
# Soumission depuis aker :
#   oarsub -l /gpu=1,walltime=12:00:00 -n run_079_speechllm_l14k_r4 \
#     /home/getalp/bonapelm/S3T/run_oar_speechllm_l14k_r4_aker.sh

#OAR -l /gpu=1,walltime=12:00:00
#OAR -n run_079_speechllm_l14k_r4
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k.yaml"
RUN_ID="run_079_aker_speechllm_l14k"
LOG="logs/run_079_train_eval.log"

echo "=== $(date -u -Iseconds) start speechLLM r4 ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
