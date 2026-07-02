#!/usr/bin/env bash
# OVH — speechLLM L-114k + Llama-3.2-3B-Instruct (run_053, piste H / B2bis).
#
# Suite run_052 (L-14k + Llama, 16,31 test) : encodeur L-114k (Phi-2 : +0,2 BLEU vs L-14k).
# Budget : ≤ 14 h GPU. Durée attendue ~3–5 h.
# Prérequis : HF_TOKEN (Llama gated + speech-large-114K).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e ssh ./ ubuntu@145.239.52.158:~/S3T/
#   ssh ubuntu@145.239.52.158 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_ovh_speechllm_114k_llama32_3b.sh \
#     > logs/run_053_speechllm_chain_wrapper.log 2>&1 &'
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_ovh_speechllm_114k_llama32_3b.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/2_speechLLM/scripts/run_053_b1_utterance_large_114k_llama32_3b_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur OVH :" >&2
    pgrep -af "^python.*pipeline\.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: aucun pipeline.py train/run Python actif."
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
  echo "WARN: HF_TOKEN absent et huggingface-cli non connecté — Llama-3.2-3B et L-114k sont gated." >&2
fi

chmod +x "$NOHUP_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM L-114k + Llama-3.2-3B (OVH, run_053) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_114k_llama32_3b.yaml \
  --run-id run_053_speechllm_b2bis_utterance_large_114k_llama32_3b \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} (max ${MAX_RUN_HOURS:-14}h) ==="
export MAX_RUN_HOURS="${MAX_RUN_HOURS:-14}"
exec bash "$NOHUP_SCRIPT"
