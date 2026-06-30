#!/usr/bin/env bash
# Modyco — speechLLM L-14k + Llama-3.2-3B-Instruct (run_052, piste H / B2bis).
#
# Budget : ≤ 14 h GPU (timeout dans le nohup). Durée attendue ~5–10 h.
# Prérequis : HF_TOKEN (modèle gated meta-llama/Llama-3.2-3B-Instruct).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e "ssh -i ~/.ssh/id_ed25519" ./ mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_speechllm_14k_llama32_3b.sh \
#     > logs/run_052_speechllm_chain_wrapper.log 2>&1 &'
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_speechllm_14k_llama32_3b.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/2_speechLLM/scripts/run_052_b1_utterance_large_14k_llama32_3b_nohup.sh"
VRAM_MAX_MIB="${VRAM_MAX_MIB:-4096}"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

gpu_vram_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '
}

require_gpu_free() {
  local vram
  vram="$(gpu_vram_used_mib || echo 99999)"
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  if [[ "${vram}" =~ ^[0-9]+$ ]] && (( vram > VRAM_MAX_MIB )); then
    echo "ERROR: VRAM occupée (${vram} MiB > ${VRAM_MAX_MIB}) :" >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -5 >&2 || true
    exit 2
  fi
  echo "OK: GPU libre (VRAM ${vram} MiB)."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,16p' "$0" | tail -n +2
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

echo "=== $(date -Is) Pré-vol speechLLM L-14k + Llama-3.2-3B (Modyco, run_052) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_llama32_3b.yaml \
  --run-id run_052_speechllm_b2bis_utterance_large_14k_llama32_3b \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} (max ${MAX_RUN_HOURS:-14}h) ==="
export MAX_RUN_HOURS="${MAX_RUN_HOURS:-14}"
exec bash "$NOHUP_SCRIPT"
