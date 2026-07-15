#!/usr/bin/env bash
# Exécuter sur aker (login) : ST L-14k v5 run_060 via OAR (~12–24 h).
set -euo pipefail
cd ~/S3T
mkdir -p logs

RUN_GLOB="run_060_aker_st_l14k_v5"
EVAL="$(ls -d runs/fr-en/${RUN_GLOB}/eval/sacrebleu_test.txt 2>/dev/null | head -1 || true)"
if [[ -n "${EVAL}" && -s "${EVAL}" ]]; then
  echo "Eval déjà présente : ${EVAL}"
  cat "${EVAL}"
  exit 0
fi

if oarstat 2>/dev/null | grep -q bonapelm; then
  echo "Job OAR bonapelm déjà actif"
  oarstat 2>/dev/null | grep bonapelm || true
  exit 0
fi

SCRIPT="${HOME}/S3T/run_oar_st_l14k_v5_aker.sh"
chmod +x "${SCRIPT}"

echo "=== $(date -Is) Attente slot OAR bonapelm ==="
while oarstat 2>/dev/null | grep -q bonapelm; do
  oarstat 2>/dev/null | grep bonapelm | head -2 || true
  sleep 120
done

echo "=== $(date -Is) oarsub run_060_st_l14k_v5 (lig-gpu10 H100 — 2080 Ti OOM) ==="
OUT="$(oarsub -S -l /gpu=1,walltime=24:00:00 -p "host='lig-gpu10.imag.fr'" -n run_060_st_l14k_v5 "${SCRIPT}" 2>&1)"
echo "${OUT}"
JOB_ID="$(echo "${OUT}" | grep -oE 'OAR_JOB_ID=[0-9]+' | cut -d= -f2)"
if [[ -z "${JOB_ID}" ]]; then
  echo "ERROR: OAR_JOB_ID introuvable" >&2
  exit 1
fi
echo "OAR_JOB_ID=${JOB_ID}" | tee logs/run_060_oarsub_id.txt
