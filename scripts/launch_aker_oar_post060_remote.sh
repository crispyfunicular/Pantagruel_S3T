#!/usr/bin/env bash
# Exécuter sur aker (login) : waiter OAR post-run_060 (077 resume → 043 ST replicate).
set -euo pipefail
cd ~/S3T
mkdir -p logs

resolve_script() {
  local name="$1"
  if [[ -f "scripts/${name}" ]]; then
    echo "scripts/${name}"
  elif [[ -f "${name}" ]]; then
    echo "${name}"
  else
    echo "ERROR: script introuvable : ${name}" >&2
    exit 2
  fi
}

is_running() {
  pgrep -af "$1" 2>/dev/null | grep -v "pgrep -af" | grep -v "bash -s" | grep -q .
}

if is_running "run_aker_wait_oar_post060.sh"; then
  echo "Waiter post060 déjà actif :"
  pgrep -af "run_aker_wait_oar_post060.sh" | grep -v pgrep || true
  tail -15 logs/run_waiter_oar_post060_aker.log 2>/dev/null || true
  exit 0
fi

if oarstat 2>/dev/null | grep -q bonapelm; then
  echo "Job OAR bonapelm déjà actif — lancement waiter quand même (reprise chaîne)"
  oarstat 2>/dev/null | grep bonapelm || true
fi

WAITER_SCRIPT="$(resolve_script run_aker_wait_oar_post060.sh)"
LAYER6_SCRIPT="$(resolve_script run_oar_speechllm_l14k_layer6_resume_aker.sh)"
REPLICATE_SCRIPT="$(resolve_script run_oar_st_l14k_v5_replicate_aker.sh)"
chmod +x "${WAITER_SCRIPT}" "${LAYER6_SCRIPT}" "${REPLICATE_SCRIPT}"

MAX_H="${MAX_HOURS:-48}"
echo "=== $(date -Is) Lancement waiter OAR post-run_060 (MAX_HOURS=${MAX_H}) ==="
nohup env MAX_HOURS="${MAX_H}" OAR_HOST="${OAR_HOST:-lig-gpu10.imag.fr}" \
  bash "${WAITER_SCRIPT}" \
  > logs/run_waiter_oar_post060_aker.log 2>&1 &
echo "WAITER_PID=$!"
sleep 5
tail -25 logs/run_waiter_oar_post060_aker.log
pgrep -af "run_aker_wait_oar_post060.sh" | grep -v pgrep || true
