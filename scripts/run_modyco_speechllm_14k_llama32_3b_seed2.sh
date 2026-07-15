#!/usr/bin/env bash
# Modyco — speechLLM L-14k + Llama-3.2-3B-Instruct seed 2 (run_055, piste F).
#
# Réplicabilité run_052 (**16,31** test, seed 42). Durée attendue ~3 h GPU.
# Prérequis : HF_TOKEN (Llama gated).
#
# Depuis le poste local :
#   rsync -az 2_speechLLM/configs/fr-en/b1_utterance_large_14k_llama32_3b_seed2.yaml \
#     2_speechLLM/scripts/run_055_b1_utterance_large_14k_llama32_3b_seed2_nohup.sh \
#     scripts/run_modyco_speechllm_14k_llama32_3b_seed2.sh \
#     -e "ssh -i ~/.ssh/id_ed25519" mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_speechllm_14k_llama32_3b_seed2.sh \
#     > logs/run_055_speechllm_chain_wrapper.log 2>&1 &'
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_speechllm_14k_llama32_3b_seed2.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/2_speechLLM/scripts/run_055_b1_utterance_large_14k_llama32_3b_seed2_nohup.sh"
VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

gpu_vram_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

require_gpu_free() {
  local vram_used vram_free
  vram_used="$(gpu_vram_used_mib || echo 99999)"
  vram_free="$(gpu_vram_free_mib || echo 0)"
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  if [[ "${vram_used}" =~ ^[0-9]+$ ]] && (( vram_used <= VRAM_MAX_MIB )); then
    echo "OK: GPU libre (VRAM ${vram_used} MiB)."
    return 0
  fi
  if [[ "${vram_free}" =~ ^[0-9]+$ ]] && (( vram_free >= 20000 )); then
    echo "OK: VRAM partielle (${vram_used} MiB used, ${vram_free} MiB free) — Llama seed2."
    return 0
  fi
  echo "ERROR: VRAM insuffisante (${vram_used} used, ${vram_free} free) :" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -5 >&2 || true
  exit 2
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

if [[ "$FORCE" != "1" ]]; then
  require_gpu_free
fi

if [[ -z "${HF_TOKEN:-}" ]] && ! huggingface-cli whoami >/dev/null 2>&1; then
  echo "WARN: HF_TOKEN absent et huggingface-cli non connecté — Llama-3.2-3B est gated." >&2
fi

chmod +x "$NOHUP_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM L-14k + Llama-3.2-3B seed 2 (Modyco, run_055) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_llama32_3b_seed2.yaml \
  --run-id run_055_speechllm_b2bis_utterance_large_14k_llama32_3b_seed2 \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} (max ${MAX_RUN_HOURS:-14}h) ==="
export MAX_RUN_HOURS="${MAX_RUN_HOURS:-14}"
exec bash "$NOHUP_SCRIPT"
