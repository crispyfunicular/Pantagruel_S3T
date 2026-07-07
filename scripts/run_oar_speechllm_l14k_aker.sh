#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k + Phi-2 seed 42 (utterance), 3e réplication IMAG.
#
# Références : run_059 **14,77** ; run_068 **13,71** ; run_070 seed 1 **15,41**.
#
# Soumission depuis aker :
#   bash -lc 'oarsub -l gpu=1,walltime=12:00:00 -n run_071_speechllm_l14k /home/getalp/bonapelm/S3T/scripts/run_oar_speechllm_l14k_aker.sh'
#
# Run cible : run_071_aker_speechllm_l14k (~6–10 h GPU sur 2080 Ti)

#OAR -l gpu=1,walltime=12:00:00
#OAR -n run_071_speechllm_l14k
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k.yaml"
RUN_ID="run_071_aker_speechllm_l14k"
LOG="logs/run_071_train_eval.log"

echo "=== $(date -u -Iseconds) start speechLLM ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
