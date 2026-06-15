#!/usr/bin/env bash
# OVH — speechLLM Pantagruel-L-114k replicate (run_032, max 48 tok).
#
# Réplication run_013 (15,24 test) après échec run_022 v3 (128 tok → 4,78).
# Durée : ~4–6 h GPU. À enchaîner après run_030 v6 long.
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e ssh ./ ubuntu@145.239.52.158:~/S3T/
#   ssh ubuntu@145.239.52.158 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_ovh_speechllm_114k_replicate.sh \
#     > logs/run_032_ovh_chain_wrapper.log 2>&1 &'
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_ovh_speechllm_114k_replicate.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NOHUP_SCRIPT="${ROOT}/2_speechLLM/scripts/run_032_b1_utterance_large_114k_replicate_nohup.sh"

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

chmod +x "$NOHUP_SCRIPT"

echo "=== $(date -Is) Pré-vol speechLLM L-114k replicate (OVH, run_032) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_114k_replicate.yaml \
  --run-id run_032_speechllm_b1_utterance_large_114k_replicate \
  --dry-run

echo "=== $(date -Is) Délégation → ${NOHUP_SCRIPT} ==="
exec bash "$NOHUP_SCRIPT"
