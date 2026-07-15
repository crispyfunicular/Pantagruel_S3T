#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k + Llama-3.2-3B, downsample k=7, max_new_tokens=128 (piste L).
#
# Job OAR IMAG — speechLLM L-14k + Llama-3.2-3B, downsample k=7 (run_067 OVH backlog).
#
# Relance après OOM 1080 Ti (OAR 128586). Demander lig-gpu10 (H100) :
#   oarsub -l /gpu=1,walltime=12:00:00 -p "host='lig-gpu10.imag.fr'" \
#     -n run_067_speechllm_llama_k7 \
#     /home/getalp/bonapelm/S3T/scripts/run_oar_speechllm_llama_k7_aker.sh
#
# Run cible : run_067_aker_speechllm_llama_k7 (~4–8 h GPU).

#OAR -l /gpu=1,walltime=12:00:00
#OAR -p host='lig-gpu10.imag.fr'
#OAR -n run_067_speechllm_llama_k7
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_llama32_3b_k7.yaml"
RUN_ID="run_067_aker_speechllm_llama_k7"
LOG="logs/run_067_speechllm_llama_k7_train_eval.log"

echo "=== $(date -u -Iseconds) start speechLLM ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
