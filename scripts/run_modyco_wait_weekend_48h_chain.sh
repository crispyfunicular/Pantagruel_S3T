#!/usr/bin/env bash
# Modyco — Chaîne waiters week-end 48 h (GPU partagé ~11 Go tiers).
#
# Ordre adapté VRAM :
#   1. run_023 Phi-2 replicate (~2 h) — tient avec ~20 Go libres
#   2. run_043 ST L-14k v5 replicate (~7–8 h)
#   3. run_066 Llama dégel (~6–14 h) — attend GPU quasi libre (< 4 Go tiers)
#   4. run_055 Llama seed 2 (~3–6 h) — idem
#
# Usage (Modyco) :
#   cd ~/S3T && mkdir -p logs
#   nohup bash scripts/run_modyco_wait_weekend_48h_chain.sh \
#     > logs/run_waiter_weekend_48h_modyco.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POLL_SEC="${POLL_SEC:-300}"
MAX_HOURS="${MAX_HOURS:-48}"
DEADLINE_EPOCH=$(( $(date +%s) + MAX_HOURS * 3600 ))
export VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"

past_deadline() {
  [[ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]
}

hours_left() {
  echo $(( (DEADLINE_EPOCH - $(date +%s)) / 3600 ))
}

wait_pipeline_idle() {
  while pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; do
    if past_deadline; then exit 0; fi
    sleep "$POLL_SEC"
  done
}

run_step() {
  local label="$1"
  local script="$2"
  echo "=== $(date -Is) ÉTAPE : ${label} (~$(hours_left)h restantes) ==="
  chmod +x "${ROOT}/scripts/${script}"
  if bash "${ROOT}/scripts/${script}"; then
    echo "=== $(date -Is) ${label} — OK ==="
  else
    echo "=== $(date -Is) ${label} — ÉCHEC (code $?) — suite chaîne ===" >&2
  fi
  wait_pipeline_idle
}

skip_if_eval() {
  local run_glob="$1"
  local eval_file
  eval_file="$(ls -d "${ROOT}/runs/fr-en/${run_glob}/eval/sacrebleu_test.txt" 2>/dev/null | head -1 || true)"
  if [[ -n "${eval_file}" && -s "${eval_file}" ]]; then
    echo "=== $(date -Is) Skip — eval déjà présent : ${eval_file} ==="
    return 0
  fi
  return 1
}

echo "=== $(date -Is) CHAÎNE week-end 48h Modyco — deadline $(date -d "@${DEADLINE_EPOCH}" -Is) ==="

# Phi-2 + ST d'abord (VRAM partielle OK)
if skip_if_eval "run_023_speechllm_b1_utterance_large_14k_replicate"; then
  :
else
  run_step "run_023 Phi-2 replicate" "run_modyco_speechllm_14k_replicate.sh"
fi
if past_deadline; then exit 0; fi

if [[ "$(hours_left)" -ge 9 ]]; then
  if skip_if_eval "run_043_transformer_baseline_utterance_large_14k_v5_replicate"; then
    :
  else
    run_step "run_043 ST v5 replicate" "run_modyco_st_14k_v5_replicate.sh"
  fi
else
  echo "=== $(date -Is) Skip run_043 — moins de 9h restantes ==="
fi
if past_deadline; then exit 0; fi

# Llama : GPU quasi libre obligatoire (pas OOM avec processus tiers)
export MIN_FREE_MIB=28000
if [[ "$(hours_left)" -ge 10 ]]; then
  run_step "run_066 Llama dégel (GPU libre)" "run_modyco_wait_gpu_then_speechllm_066_llama_unfreeze.sh"
else
  echo "=== $(date -Is) Skip run_066 — moins de 10h restantes ==="
fi
if past_deadline; then exit 0; fi

export MIN_FREE_MIB=28000
if [[ "$(hours_left)" -ge 8 ]]; then
  run_step "run_055 Llama seed 2 (GPU libre)" "run_modyco_wait_gpu_then_speechllm_055_llama_seed2.sh"
else
  echo "=== $(date -Is) Skip run_055 — moins de 8h restantes ==="
fi

echo "=== $(date -Is) CHAÎNE week-end 48h Modyco TERMINÉE ==="
