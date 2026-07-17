#!/usr/bin/env bash
# Job OAR IMAG — éval seule run_084 (best.pt déjà entraîné).
#
# Le train a réussi ; l'éval a échoué (401 HF / token OAuth invalide).
# On force le mode offline : Llama-3.2-3B est déjà en cache HF sur aker.
#
#   oarsub -S -l /gpu=1,walltime=03:00:00 -p "host='lig-gpu10.imag.fr'" \
#     -n run_084_eval_seed2 \
#     ~/S3T/scripts/run_oar_eval_084_seed2_aker.sh

#OAR -l /gpu=1,walltime=03:00:00
#OAR -p host='lig-gpu10.imag.fr'
#OAR -n run_084_eval_seed2
set -euo pipefail

cd "${HOME}/S3T"
source .venv/bin/activate
mkdir -p logs

# Token HF aker cassé (signature OAuth) — utiliser le cache local uniquement.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CFG="2_speechLLM/configs/fr-en/b2_utterance_large_14k_llama32_3b_unfreeze_seed2.yaml"
RUN_ID="run_084_aker_speechllm_b2_utterance_large_14k_llama32_3b_unfreeze_seed2"
LOG="logs/run_084_aker_llama_unfreeze_seed2_eval_only.log"
CKPT="runs/fr-en/${RUN_ID}/checkpoints/best.pt"

echo "=== $(date -u -Iseconds) start EVAL-ONLY ${RUN_ID} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint absent: ${CKPT}" >&2
  exit 2
fi

: > "${LOG}"
python 2_speechLLM/pipeline.py evaluate \
  --config "${CFG}" \
  --run-id "${RUN_ID}" \
  --checkpoint "${CKPT}" \
  -v 2>&1 | tee -a "${LOG}"

echo "=== $(date -u -Iseconds) done exit=$? ==="
