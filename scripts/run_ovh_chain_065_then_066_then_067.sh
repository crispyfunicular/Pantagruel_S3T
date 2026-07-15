#!/usr/bin/env bash
# OVH — Chaîne backlog juillet 2026 :
#   1. run_065 : ST L-14k batch 32 safe (piste B)
#   2. run_066 : speechLLM Llama dégel encodeur (piste L)
#   3. run_067 : speechLLM Llama k=7 + 128 tokens (piste L)
#
# Total estimé : ~18–32 h GPU séquentiel.
#
# Lancement direct (GPU libre) :
#   nohup bash scripts/run_ovh_chain_065_then_066_then_067.sh \
#     > logs/run_chain_065_067_wrapper.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

SCRIPT_065="${ROOT}/scripts/run_ovh_st_065_l14k_batch32_safe.sh"
SCRIPT_066="${ROOT}/scripts/run_ovh_speechllm_066_llama_unfreeze.sh"
SCRIPT_067="${ROOT}/scripts/run_ovh_speechllm_067_llama_k7.sh"

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: GPU libre (pas de train/run actif)."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,18p' "$0" | tail -n +2
      exit 0
      ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      exit 2
      ;;
  esac
done

[[ "$FORCE" != "1" ]] && require_gpu_free
mkdir -p logs

bleu_from_run() {
  local run_id="$1"
  local path="runs/fr-en/${run_id}/eval/sacrebleu_test.txt"
  if [[ -f "$path" ]]; then
    head -1 "$path"
  else
    echo "n/a"
  fi
}

echo "=== $(date -Is) [1/3] LANCEMENT run_065 ST batch32 safe ==="
bash "$SCRIPT_065"
BLEU_065="$(bleu_from_run run_065_transformer_baseline_utterance_large_14k_v13_batch32_safe)"
echo "=== $(date -Is) [1/3] run_065 — BLEU test : ${BLEU_065} ==="

echo "=== $(date -Is) [2/3] LANCEMENT run_066 speechLLM Llama dégel ==="
bash "$SCRIPT_066"
BLEU_066="$(bleu_from_run run_066_speechllm_b2_utterance_large_14k_llama32_3b_unfreeze)"
echo "=== $(date -Is) [2/3] run_066 — BLEU test : ${BLEU_066} ==="

echo "=== $(date -Is) [3/3] LANCEMENT run_067 speechLLM Llama k=7 ==="
bash "$SCRIPT_067"
BLEU_067="$(bleu_from_run run_067_speechllm_b1_utterance_large_14k_llama32_3b_k7)"
echo "=== $(date -Is) [3/3] run_067 — BLEU test : ${BLEU_067} ==="

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  CHAÎNE OVH 065→066→067 TERMINÉE                                   ║"
echo "╠══════════════════════════════════════════════════════════════════════╣"
echo "║  run_065  ST L-14k batch32 safe     : ${BLEU_065}"
echo "║  run_066  Llama dégel encodeur      : ${BLEU_066}"
echo "║  run_067  Llama k=7 + 128 tokens    : ${BLEU_067}"
echo "╚══════════════════════════════════════════════════════════════════════╝"
