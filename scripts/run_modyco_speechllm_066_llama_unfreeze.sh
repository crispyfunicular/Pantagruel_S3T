#!/usr/bin/env bash
# Modyco — speechLLM L-14k + Llama-3.2-3B, encodeur dégelé (run_066, piste L).
#
# Réplication Modyco de run_066 IMAG (**20,12** test). LR encodeur 1e-5.
# Budget : ≤ 14 h GPU. Durée attendue ~6–12 h (tour partagée : préférer GPU libre).
# Prérequis : HF_TOKEN ou huggingface-cli login (Llama gated).
#
# Usage (Modyco, GPU libre) :
#   cd ~/S3T && source .venv/bin/activate && mkdir -p logs
#   nohup bash scripts/run_modyco_speechllm_066_llama_unfreeze.sh \
#     > logs/run_066_modyco_launch.log 2>&1 &
#
# Via waiter :
#   nohup bash scripts/run_modyco_wait_gpu_then_speechllm_066_llama_unfreeze.sh \
#     > logs/run_waiter_066_modyco.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

MAX_HOURS_SLM="${MAX_HOURS_SLM:-14}"
VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"
MIN_FREE_MIB="${MIN_FREE_MIB:-20000}"

CFG="2_speechLLM/configs/fr-en/b2_utterance_large_14k_llama32_3b_unfreeze.yaml"
RUN_ID="run_066_modyco_speechllm_b2_utterance_large_14k_llama32_3b_unfreeze"
LOG="logs/${RUN_ID}_train_eval.log"

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_vram_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

require_gpu_ok() {
  local vram_used vram_free
  vram_used="$(gpu_vram_used_mib || echo 99999)"
  vram_free="$(gpu_vram_free_mib || echo 0)"
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  if [[ "${1:-}" == "--force" ]]; then
    echo "OK: --force (VRAM used ${vram_used} MiB, free ${vram_free} MiB)."
    return 0
  fi
  if [[ "${vram_used}" =~ ^[0-9]+$ ]] && (( vram_used <= VRAM_MAX_MIB )); then
    echo "OK: GPU libre (VRAM ${vram_used} MiB)."
    return 0
  fi
  if [[ "${vram_free}" =~ ^[0-9]+$ ]] && (( vram_free >= MIN_FREE_MIB )); then
    echo "OK: VRAM partielle (${vram_used} MiB used, ${vram_free} MiB free) — entraînement Llama."
    return 0
  fi
  echo "ERROR: VRAM insuffisante pour Llama dégelé (used ${vram_used}, free ${vram_free}, need ${MIN_FREE_MIB}) :" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -5 >&2 || true
  exit 2
}

FORCE=0
OVERWRITE="${OVERWRITE:-0}"
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --overwrite) OVERWRITE=1 ;;
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

OVERWRITE_ARGS=()
if [[ "$OVERWRITE" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
  echo "=== $(date -Is) Repart de zéro (--overwrite) ==="
fi

[[ "$FORCE" == "1" ]] && require_gpu_ok --force || require_gpu_ok
mkdir -p logs

if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "WARN: HF_TOKEN absent — Llama-3.2-3B est gated." >&2
fi

echo "=== $(date -Is) Dry-run ${RUN_ID} ==="
python 2_speechLLM/pipeline.py train --config "$CFG" --run-id "$RUN_ID" --dry-run

echo "=== $(date -Is) LANCEMENT ${RUN_ID} (max ${MAX_HOURS_SLM}h) ==="
{
  timeout "$((MAX_HOURS_SLM * 3600))" \
    python 2_speechLLM/pipeline.py run \
      --config "$CFG" \
      --run-id "$RUN_ID" \
      "${OVERWRITE_ARGS[@]}" \
      -v
  ec=$?
  if [[ "$ec" -eq 124 ]]; then
    echo "=== $(date -Is) TIMEOUT ${MAX_HOURS_SLM}h — éval best checkpoint ==="
    python 2_speechLLM/pipeline.py evaluate \
      --config "$CFG" \
      --run-id "$RUN_ID" -v || true
  elif [[ "$ec" -ne 0 ]]; then
    exit "$ec"
  fi
} 2>&1 | tee -a "$LOG"

BLEU="n/a"
[[ -f "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt" ]] \
  && BLEU="$(head -1 "runs/fr-en/${RUN_ID}/eval/sacrebleu_test.txt")"
echo "=== $(date -Is) ${RUN_ID} — BLEU test : ${BLEU} ==="
