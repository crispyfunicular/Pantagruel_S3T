#!/usr/bin/env bash
# Modyco — Waiter GPU 13 h max : run_044 L-114k SpecAug → run_046 v11 batch32 (piste B).
# run_046 est relancé si eval présent mais BLEU test < MIN_BLEU (défaut 15).
#
# Budget total : 13 h (deadline absolue). GPU partagé : poll jusqu'à VRAM ≤ 4 Go.
#
# Usage (Modyco) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_modyco_wait_gpu_13h_chain.sh \
#     > logs/run_waiter_13h_modyco.log 2>&1 &
#   tail -f logs/run_waiter_13h_modyco.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

POLL_SEC="${POLL_SEC:-300}"
MAX_HOURS="${MAX_HOURS:-13}"
DEADLINE_EPOCH=$(( $(date +%s) + MAX_HOURS * 3600 ))
VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"
MIN_BLEU="${MIN_BLEU:-15}"

past_deadline() {
  [[ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]
}

hours_left() {
  echo $(( (DEADLINE_EPOCH - $(date +%s)) / 3600 ))
}

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

wait_gpu_available() {
  echo "=== $(date -Is) Attente GPU libre (poll ${POLL_SEC}s, ~$(hours_left)h restantes) ==="
  while true; do
    if past_deadline; then
      echo "=== $(date -Is) DEADLINE ${MAX_HOURS}h — arrêt waiter ==="
      exit 0
    fi
    local vram
    vram="$(gpu_vram_used_mib || echo 99999)"
    if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
      pgrep -af "^python.*pipeline\.py (train|run)" | head -1 || true
      echo "$(date -Is) pipeline S3T actif — attente ${POLL_SEC}s"
    elif [[ "${vram}" =~ ^[0-9]+$ ]] && (( vram > VRAM_MAX_MIB )); then
      echo "$(date -Is) VRAM occupée (${vram} MiB > ${VRAM_MAX_MIB}) — attente ${POLL_SEC}s"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -3 || true
    else
      echo "=== $(date -Is) GPU libre (VRAM ${vram} MiB) ==="
      return 0
    fi
    sleep "$POLL_SEC"
  done
}

wait_pipeline_idle() {
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    if past_deadline; then
      return 1
    fi
    sleep "$POLL_SEC"
  done
}

eval_bleu_score() {
  local eval_file="$1"
  local line bleu
  line="$(head -1 "${eval_file}" 2>/dev/null || true)"
  bleu="$(echo "${line}" | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
  if [[ -n "${bleu}" ]]; then
    echo "${bleu}"
    return 0
  fi
  return 1
}

skip_if_eval() {
  local run_glob="$1"
  local min_bleu="${2:-}"
  local eval_file bleu
  eval_file="$(ls -d "${ROOT}/runs/fr-en/${run_glob}/eval/sacrebleu_test.txt" 2>/dev/null | head -1 || true)"
  if [[ -z "${eval_file}" || ! -s "${eval_file}" ]]; then
    return 1
  fi
  if [[ -n "${min_bleu}" ]]; then
    bleu="$(eval_bleu_score "${eval_file}" || true)"
    if [[ -n "${bleu}" ]] && awk -v b="${bleu}" -v m="${min_bleu}" 'BEGIN { exit !(b+0 >= m+0) }'; then
      echo "=== $(date -Is) Skip — eval OK (BLEU ${bleu} ≥ ${min_bleu}) : ${eval_file} ==="
      head -1 "${eval_file}" || true
      return 0
    fi
    echo "=== $(date -Is) Eval insuffisant (BLEU ${bleu:-?} < ${min_bleu}) — relance : ${eval_file} ==="
    return 1
  fi
  echo "=== $(date -Is) Skip — eval déjà présent : ${eval_file} ==="
  head -1 "${eval_file}" || true
  return 0
}

run_step() {
  local label="$1"
  local script="$2"
  local min_h="${3:-1}"
  shift 3 || true
  echo "=== $(date -Is) ÉTAPE : ${label} (~$(hours_left)h restantes) ==="
  if [[ "$(hours_left)" -lt "${min_h}" ]]; then
    echo "=== $(date -Is) Moins de ${min_h}h restantes — skip ${label} ==="
    return 0
  fi
  wait_gpu_available
  if past_deadline; then
    return 0
  fi
  chmod +x "${ROOT}/scripts/${script}"
  if bash "${ROOT}/scripts/${script}" "$@"; then
    echo "=== $(date -Is) ${label} — OK ==="
  else
    echo "=== $(date -Is) ${label} — ÉCHEC (code $?) — suite chaîne ===" >&2
  fi
  wait_pipeline_idle || return 1
}

echo "=== $(date -Is) WAITER GPU ${MAX_HOURS}h Modyco — deadline $(date -d "@${DEADLINE_EPOCH}" -Is) ==="

if skip_if_eval "run_044_speechllm_b1_utterance_large_114k_v5_specaug"; then
  :
else
  run_step "run_044 L-114k SpecAug (~3 h)" "run_modyco_speechllm_114k_v5_specaug.sh" 3 || exit 0
fi

if past_deadline; then
  echo "=== $(date -Is) DEADLINE atteinte après run_044 ==="
  exit 0
fi

if skip_if_eval "run_046_transformer_baseline_utterance_large_14k_v11_batch32" "${MIN_BLEU}"; then
  :
else
  run_step "run_046 v11 batch32 (~8–10 h)" "run_modyco_st_14k_v11_batch32.sh" 7 --overwrite || exit 0
fi

echo "=== $(date -Is) WAITER GPU ${MAX_HOURS}h Modyco TERMINÉ ==="
