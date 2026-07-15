#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k + Phi-2, couche encodeur 9 (piste J).
#
# Référence : run_012 (dernière couche **15,03**) ; run_056 L-114k layer9 OVH **14,52**.
# Seul changement vs baseline IMAG (run_059 **14,77**) : encoder_layer=9.
# Durée attendue : ~6–10 h GPU sur 2080 Ti (20k updates).
#
# Soumission depuis aker (login shell) :
#   oarsub -l /gpu=1,walltime=12:00:00 -n run_076_speechllm_l14k_layer9 \
#     /home/getalp/bonapelm/S3T/scripts/run_oar_speechllm_l14k_layer9_aker.sh

#OAR -l /gpu=1,walltime=12:00:00
#OAR -n run_076_speechllm_l14k_layer9
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_layer9.yaml"
RUN_ID="run_076_aker_speechllm_l14k_layer9"
LOG="logs/run_076_train_eval.log"

echo "=== $(date -u -Iseconds) start speechLLM layer9 ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
