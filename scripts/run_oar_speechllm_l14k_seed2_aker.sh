#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k + Phi-2, seed 1 (piste F).
#
# Relance run_070 (walltime 12 h, 6 juil.) — reprise depuis checkpoints/best.pt.
# Référence Modyco : run_050 (**14,01** test). Walltime max OAR : **12 h**.
#
# Soumission depuis aker :
#   oarsub -l gpu=1,walltime=12:00:00 -n run_070_resume_speechllm_l14k_seed2 \
#     /home/getalp/bonapelm/S3T/scripts/run_oar_speechllm_l14k_seed2_aker.sh
#
# Run cible : run_070_aker_speechllm_l14k_seed2 (~6–12 h GPU)

#OAR -n run_070_speechllm_l14k_seed2
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_seed2.yaml"
RUN_ID="run_070_aker_speechllm_l14k_seed2"
LOG="logs/run_070_resume_train_eval.log"

echo "=== $(date -u -Iseconds) resume speechLLM ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  --resume \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
