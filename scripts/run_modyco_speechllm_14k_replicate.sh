#!/usr/bin/env bash
# Modyco — speechLLM L-14k réplication run_012 (run_023, max 48 tok).
#
# Budget : ≤ 3 h GPU. Durée attendue ~1,5–2 h.
# Objectif : reproduire run_012 OVH (15,03 test) après échec run_021 (128 tok → 5,48).
#
# Depuis le poste local :
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_speechllm_14k_replicate.sh \
#     > logs/run_023_speechllm_chain_wrapper.log 2>&1 &'
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_speechllm_14k_replicate.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/2_speechLLM/scripts/run_023_b1_utterance_large_14k_replicate_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "^python.*pipeline\.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
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
      sed -n '1,14p' "$0" | tail -n +2
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

chmod +x "$NOHUP_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM L-14k replicate (Modyco, run_023) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_replicate.yaml \
  --run-id run_023_speechllm_b1_utterance_large_14k_replicate \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} ==="
exec bash "$NOHUP_SCRIPT"
