#!/usr/bin/env bash
# Job OAR IMAG — speechLLM L-14k + Llama-3.2-3B, encodeur dégelé (run_066 OVH backlog).
#
# Miroir OVH : scripts/run_ovh_speechllm_066_llama_unfreeze.sh (~6–12 h GPU).
# OOM sur 1080 Ti (OAR 129787) — cibler lig-gpu10 (H100) obligatoire.
#
# Soumission depuis aker (login shell) :
#   oarsub -S -l /gpu=1,walltime=12:00:00 -p "host='lig-gpu10.imag.fr'" \
#     -n run_084_speechllm_llama_unfreeze_seed2 \
#     ~/S3T/scripts/run_oar_speechllm_066_llama_unfreeze_seed2_aker.sh

#OAR -l /gpu=1,walltime=12:00:00
#OAR -p host='lig-gpu10.imag.fr'
#OAR -n run_084_speechllm_llama_unfreeze_seed2
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

CFG="2_speechLLM/configs/fr-en/b2_utterance_large_14k_llama32_3b_unfreeze_seed2.yaml"
RUN_ID="run_084_aker_speechllm_b2_utterance_large_14k_llama32_3b_unfreeze_seed2"
LOG="logs/run_084_aker_llama_unfreeze_seed2_train_eval.log"

echo "=== $(date -u -Iseconds) start speechLLM ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "WARN: HF_TOKEN absent — Llama-3.2-3B est gated." >&2
fi

# Log frais à chaque soumission OAR (évite de mélanger avec un échec OOM précédent).
: > "${LOG}"

python 2_speechLLM/pipeline.py run \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
