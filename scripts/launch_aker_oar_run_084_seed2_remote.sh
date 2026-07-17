#!/usr/bin/env bash
# Exécuter sur aker (login) : speechLLM run_084 seed2 via OAR sur lig-gpu10 (H100).
set -euo pipefail
cd ~/S3T
mkdir -p logs

OAR_HOST="${OAR_HOST:-lig-gpu10.imag.fr}"
WALLTIME="${WALLTIME:-12:00:00}"

RUN_GLOB="run_084_aker_speechllm_b2_utterance_large_14k_llama32_3b_unfreeze_seed2"
EVAL="$(ls -d runs/fr-en/${RUN_GLOB}/eval/sacrebleu_test.txt 2>/dev/null | head -1 || true)"
if [[ -n "${EVAL}" && -s "${EVAL}" ]]; then
  echo "Eval déjà présente : ${EVAL}"
  cat "${EVAL}"
  exit 0
fi

if oarstat 2>/dev/null | grep -q bonapelm; then
  echo "Job OAR bonapelm déjà actif :"
  oarstat 2>/dev/null | grep bonapelm || true
  exit 0
fi

SCRIPT="${HOME}/S3T/scripts/run_oar_speechllm_066_llama_unfreeze_seed2_aker.sh"
chmod +x "${SCRIPT}"

echo "=== $(date -Is) Attente slot OAR bonapelm (poll 120s) ==="
while oarstat 2>/dev/null | grep -q bonapelm; do
  oarstat 2>/dev/null | grep bonapelm | head -2 || true
  sleep 120
done

echo "=== $(date -Is) oarsub run_084_speechllm_llama_unfreeze_seed2 (host=${OAR_HOST} H100) ==="
OUT="$(oarsub -S -l "/gpu=1,walltime=${WALLTIME}" -p "host='${OAR_HOST}'" \
  -n run_084_speechllm_llama_unfreeze_seed2 "${SCRIPT}" 2>&1)"
echo "${OUT}" | tee -a logs/run_084_launch.log
JOB_ID="$(echo "${OUT}" | grep -oE 'OAR_JOB_ID=[0-9]+' | cut -d= -f2)"
if [[ -z "${JOB_ID}" ]]; then
  echo "ERROR: OAR_JOB_ID introuvable" >&2
  exit 1
fi
echo "OAR_JOB_ID=${JOB_ID}" | tee logs/run_084_oarsub_id.txt
