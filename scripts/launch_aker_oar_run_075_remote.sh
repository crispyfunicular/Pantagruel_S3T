#!/usr/bin/env bash
# Exécuter sur aker (login) : open baselines Whisper + Seamless (run_075, eval ~2–4 h).
set -euo pipefail
cd ~/S3T
mkdir -p logs

WHISPER="runs/fr-en/run_075_aker_open_whisper_st_utterance/eval/sacrebleu_test.txt"
SEAMLESS="runs/fr-en/run_075b_aker_open_seamlessm4t_v2_utterance/eval/sacrebleu_test.txt"
if [[ -s "${WHISPER}" && -s "${SEAMLESS}" ]]; then
  echo "Evals déjà présentes"
  head -1 "${WHISPER}" "${SEAMLESS}"
  exit 0
fi

if oarstat 2>/dev/null | grep -q bonapelm; then
  echo "Job OAR bonapelm déjà actif"
  oarstat 2>/dev/null | grep bonapelm || true
  exit 0
fi

SCRIPT="${HOME}/S3T/run_oar_open_baselines_whisper_seamless_aker.sh"
chmod +x "${SCRIPT}"

echo "=== $(date -Is) Attente slot OAR bonapelm ==="
while oarstat 2>/dev/null | grep -q bonapelm; do
  oarstat 2>/dev/null | grep bonapelm | head -2 || true
  sleep 120
done

echo "=== $(date -Is) oarsub run_075_open_baselines_aker ==="
OUT="$(oarsub -l /gpu=1,walltime=12:00:00 -n run_075_open_baselines_aker "${SCRIPT}" 2>&1)"
echo "${OUT}"
JOB_ID="$(echo "${OUT}" | grep -oE 'OAR_JOB_ID=[0-9]+' | cut -d= -f2)"
if [[ -z "${JOB_ID}" ]]; then
  echo "ERROR: OAR_JOB_ID introuvable" >&2
  exit 1
fi
echo "OAR_JOB_ID=${JOB_ID}" | tee logs/run_075_oarsub_id.txt
