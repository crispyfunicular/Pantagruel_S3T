#!/usr/bin/env bash
# Modyco — speechLLM L-14k + Mistral-7B-Instruct 4-bit (run_054, piste H / B2bis).
#
# Suite run_052 (Llama @ 16,31 test) : dernier LLM B2bis sur encodeur L-14k.
# Budget : ≤ 14 h GPU. Durée attendue ~4–8 h.
# Prérequis : bitsandbytes installé (`pip install -r requirements.txt`).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e "ssh -i ~/.ssh/id_ed25519" ./ mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_speechllm_14k_mistral_7b.sh \
#     > logs/run_054_speechllm_chain_wrapper.log 2>&1 &'
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_speechllm_14k_mistral_7b.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/2_speechLLM/scripts/run_054_b1_utterance_large_14k_mistral_7b_nohup.sh"
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

if ! python -c "import bitsandbytes" >/dev/null 2>&1; then
  echo "ERROR: bitsandbytes requis pour Mistral 7B 4-bit — pip install -r requirements.txt" >&2
  exit 2
fi

chmod +x "$NOHUP_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM L-14k + Mistral-7B 4-bit (Modyco, run_054) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_mistral_7b.yaml \
  --run-id run_054_speechllm_b2bis_utterance_large_14k_mistral_7b \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} (max ${MAX_RUN_HOURS:-14}h) ==="
export MAX_RUN_HOURS="${MAX_RUN_HOURS:-14}"
exec bash "$NOHUP_SCRIPT"
