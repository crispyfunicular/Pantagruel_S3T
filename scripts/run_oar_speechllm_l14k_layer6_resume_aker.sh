#!/usr/bin/env bash
# Job OAR IMAG — reprise speechLLM L-14k couche 6 (run_077, piste J).
#
# Première passe OAR 128866 : KILLED walltime 12 h @ ~11,3k/20k updates (best dev ~8,93).
# Reprise OAR 129041 : échec immédiat — pipeline.py aker sans `--resume` (code non déployé).
#
# Soumission depuis aker (login shell) :
#   oarsub -l /gpu=1,walltime=16:00:00 -n run_077_resume_speechllm_l14k_layer6 \
#     /home/getalp/bonapelm/S3T/run_oar_speechllm_l14k_layer6_resume_aker.sh

#OAR -l /gpu=1,walltime=16:00:00
#OAR -n run_077_resume_speechllm_l14k_layer6
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b1_utterance_large_14k_layer6.yaml"
RUN_ID="run_077_aker_speechllm_l14k_layer6"
LOG="logs/run_077_resume_train_eval.log"

echo "=== $(date -u -Iseconds) resume speechLLM layer6 ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  --resume \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
