#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k + Llama-3.2-3B, encodeur dégelé (run_066 OVH backlog).
#
# Miroir OVH : scripts/run_ovh_speechllm_066_llama_unfreeze.sh (~6–12 h GPU).
# Attention : OOM possible sur 2080 Ti 11 Go — privilégier nœud H100 si admission OAR le permet.
#
# Soumission depuis aker (login shell) :
#   oarsub -l /gpu=1,walltime=12:00:00 -n run_066_speechllm_llama_unfreeze \
#     /home/getalp/bonapelm/S3T/scripts/run_oar_speechllm_066_llama_unfreeze_aker.sh

#OAR -l /gpu=1,walltime=12:00:00
#OAR -n run_066_speechllm_llama_unfreeze
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b2_utterance_large_14k_llama32_3b_unfreeze.yaml"
RUN_ID="run_066_aker_speechllm_b2_utterance_large_14k_llama32_3b_unfreeze"
LOG="logs/run_066_aker_llama_unfreeze_train_eval.log"

echo "=== $(date -u -Iseconds) start speechLLM ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "WARN: HF_TOKEN absent — Llama-3.2-3B est gated." >&2
fi

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
