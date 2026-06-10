#!/usr/bin/env bash
# Modyco — speechLLM B1 Pantagruel-L-14k dégelé (run_015).
#
# Suite logique après ST run_014 terminé (~17 BLEU) et speechLLM run_012 (15 BLEU gelé).
# Pistes variantes.md §2 : encodeur 14k dégelé + max_new_tokens 128 + early stop.
# Budget : ≤ 4 h GPU (estimé 2–3,5 h).
#
# Depuis le poste local :
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_speechllm_14k_unfreeze.sh \
#     > logs/run_015_speechllm_unfreeze_chain_wrapper.log 2>&1 &'
#
# Sur Modyco :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_speechllm_14k_unfreeze.sh \
#     > logs/run_015_speechllm_unfreeze_chain_wrapper.log 2>&1 &
#   tail -f logs/run_015_speechllm_b1_utterance_large_14k_unfreeze_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_speechllm_14k_unfreeze.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPEECHLLM_SCRIPT="${ROOT}/2_speechLLM/scripts/run_015_b1_utterance_large_14k_unfreeze_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "pipeline.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "pipeline.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: aucun pipeline.py train/run actif."
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help)
      sed -n '1,22p' "$0" | tail -n +2
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

chmod +x "$SPEECHLLM_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM L-14k unfreeze (Modyco, run_015) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_unfreeze.yaml \
  --run-id run_015_speechllm_b1_utterance_large_14k_unfreeze \
  --dry-run

echo "=== $(date -Is) Délégation → ${SPEECHLLM_SCRIPT} ==="
exec bash "$SPEECHLLM_SCRIPT"
