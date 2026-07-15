#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k + Phi-2, couche encodeur 6 (piste J).
#
# Réplication IMAG de run_048 Modyco (**12,41** test) ; complète run_076 layer9 (**12,06**).
# Durée attendue : ~6–10 h GPU sur 2080 Ti (20k updates).
#
# Soumission depuis aker (login shell) :
#   oarsub -l /gpu=1,walltime=12:00:00 -n run_077_speechllm_l14k_layer6 \
#     /home/getalp/bonapelm/S3T/run_oar_speechllm_l14k_layer6_aker.sh

#OAR -l /gpu=1,walltime=12:00:00
#OAR -n run_077_speechllm_l14k_layer6
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_layer6.yaml"
RUN_ID="run_077_aker_speechllm_l14k_layer6"
LOG="logs/run_077_train_eval.log"

echo "=== $(date -u -Iseconds) start speechLLM layer6 ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
