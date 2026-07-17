#!/usr/bin/env bash
# aker (login) — Waiter OAR post-run_060 : run_077 layer6 resume → run_043 ST replicate.
#
# Tour partagée IMAG : un seul job OAR bonapelm à la fois.
# Budget total : 48 h (deadline absolue) — ajustable via MAX_HOURS.
#
# Usage (aker, login shell) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_aker_wait_oar_post060.sh \
#     > logs/run_waiter_oar_post060_aker.log 2>&1 &
#   tail -f logs/run_waiter_oar_post060_aker.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "${SCRIPT_DIR}/2_speechLLM/pipeline.py" ]]; then
  ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../2_speechLLM/pipeline.py" ]]; then
  ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "ERROR: racine S3T introuvable depuis ${SCRIPT_DIR}" >&2
  exit 2
fi
cd "$ROOT"

script_path() {
  local name="$1"
  if [[ -f "${ROOT}/scripts/${name}" ]]; then
    echo "${ROOT}/scripts/${name}"
  elif [[ -f "${ROOT}/${name}" ]]; then
    echo "${ROOT}/${name}"
  else
    echo "ERROR: script introuvable : ${name} (scripts/ ou racine S3T)" >&2
    return 1
  fi
}

POLL_SEC="${POLL_SEC:-300}"
MAX_HOURS="${MAX_HOURS:-48}"
DEADLINE_EPOCH=$(( $(date +%s) + MAX_HOURS * 3600 ))
OAR_HOST="${OAR_HOST:-lig-gpu10.imag.fr}"

OAR_JOBS=(
  "run_077_resume_speechllm_l14k_layer6|16:00:00|run_oar_speechllm_l14k_layer6_resume_aker.sh"
  "run_043_st_l14k_v5_replicate|12:00:00|run_oar_st_l14k_v5_replicate_aker.sh"
)

past_deadline() {
  [[ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]
}

hours_left() {
  echo $(( (DEADLINE_EPOCH - $(date +%s)) / 3600 ))
}

wait_oar_slot() {
  echo "=== $(date -Is) Attente slot OAR bonapelm (poll ${POLL_SEC}s, ~$(hours_left)h restantes) ==="
  while oarstat 2>/dev/null | grep -q bonapelm; do
    if past_deadline; then
      echo "=== $(date -Is) DEADLINE ${MAX_HOURS}h — arrêt waiter OAR ==="
      exit 0
    fi
    oarstat 2>/dev/null | grep bonapelm | head -2 || true
    sleep "$POLL_SEC"
  done
}

submit_oar() {
  local name="$1"
  local walltime="$2"
  local script="$3"
  chmod +x "$script"
  echo "=== $(date -Is) oarsub ${name} walltime=${walltime} host=${OAR_HOST} ===" >&2
  local out job_id oarsub_args=(-S -l "/gpu=1,walltime=${walltime}" -n "${name}")
  if [[ -n "${OAR_HOST}" ]]; then
    oarsub_args+=(-p "host='${OAR_HOST}'")
  fi
  oarsub_args+=("${script}")
  out="$(oarsub "${oarsub_args[@]}" 2>&1)"
  echo "$out" >&2
  job_id="$(echo "$out" | grep -oE 'OAR_JOB_ID=[0-9]+' | cut -d= -f2)"
  if [[ -z "${job_id}" ]]; then
    echo "ERROR: oarsub ${name} — OAR_JOB_ID introuvable" >&2
    return 1
  fi
  echo "${job_id}"
}

wait_oar_job() {
  local job_id="$1"
  echo "=== $(date -Is) Attente fin OAR ${job_id} ==="
  while true; do
    local line
    line="$(oarstat -j "${job_id}" 2>/dev/null | grep -E "^[[:space:]]*${job_id}[[:space:]]" || true)"
    if [[ -z "${line}" ]]; then
      break
    fi
    if echo "${line}" | grep -qE '[[:space:]][RW][[:space:]]'; then
      if past_deadline; then
        echo "=== $(date -Is) DEADLINE atteinte pendant job ${job_id} ==="
        return 1
      fi
      echo "${line}"
      sleep "$POLL_SEC"
      continue
    fi
    echo "${line}"
    break
  done
  echo "=== $(date -Is) OAR ${job_id} terminé ==="
}

skip_if_eval() {
  local run_glob="$1"
  local eval_file
  eval_file="$(ls -d "${ROOT}/runs/fr-en/${run_glob}/eval/sacrebleu_test.txt" 2>/dev/null | head -1 || true)"
  if [[ -n "${eval_file}" && -s "${eval_file}" ]]; then
    echo "=== $(date -Is) Skip OAR — eval déjà présent : ${eval_file} ==="
    head -1 "${eval_file}" || true
    return 0
  fi
  return 1
}

wait_existing_oar() {
  local name="$1"
  local job_line job_id
  job_line="$(oarstat 2>/dev/null | grep bonapelm | grep "${name}" | head -1 || true)"
  if [[ -z "${job_line}" ]]; then
    return 1
  fi
  job_id="$(echo "${job_line}" | awk '{print $1}')"
  echo "=== $(date -Is) Job OAR déjà actif pour ${name} : ${job_id} ==="
  wait_oar_job "${job_id}"
  return 0
}

echo "=== $(date -Is) WAITER OAR post-run_060 aker — deadline $(date -d "@${DEADLINE_EPOCH}" -Is) ==="

for entry in "${OAR_JOBS[@]}"; do
  IFS='|' read -r name walltime script_name <<< "$entry"
  script="$(script_path "${script_name}")"
  case "${name}" in
    run_077_resume_speechllm_l14k_layer6)
      skip_if_eval "run_077_aker_speechllm_l14k_layer6" && continue
      wait_existing_oar "${name}" && continue
      ;;
    run_043_st_l14k_v5_replicate)
      skip_if_eval "run_043_aker_st_l14k_v5_replicate" && continue
      wait_existing_oar "${name}" && continue
      ;;
  esac
  min_h="${walltime%%:*}"
  if [[ "$(hours_left)" -lt "${min_h}" ]]; then
    echo "=== $(date -Is) Moins de ${min_h}h restantes — skip ${name} ==="
    continue
  fi
  wait_oar_slot
  if past_deadline; then
    break
  fi
  job_id="$(submit_oar "$name" "$walltime" "$script")"
  [[ -n "${job_id}" ]] || { echo "ERROR: oarsub ${name} échoué" >&2; continue; }
  wait_oar_job "${job_id}" || break
done

echo "=== $(date -Is) WAITER OAR post-run_060 TERMINÉ ==="
