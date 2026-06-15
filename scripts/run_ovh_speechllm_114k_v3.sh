#!/usr/bin/env bash
# OVH — speechLLM Pantagruel-L-114k v3 (run_022).
#
# Suite à run_013 (15,24 test) et échec run_017 v2 (5,60 test).
# Calqué sur run_021 L-14k v3 (Modyco) : eval dev complet pour early stop fiable.
# Durée : ~4–6 h GPU (early stop possible avant 20k).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e ssh ./ ubuntu@145.239.52.158:~/S3T/
#   ssh ubuntu@145.239.52.158 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_ovh_speechllm_114k_v3.sh > logs/run_022_ovh_chain_wrapper.log 2>&1 &'
#
# Enchaînement après run_019 ST :
#   nohup bash scripts/run_ovh_wait_st_then_speechllm_114k_v3.sh \
#     > logs/run_022_ovh_wait_chain.log 2>&1 &
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_ovh_speechllm_114k_v3.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPEECHLLM_SCRIPT="${ROOT}/2_speechLLM/scripts/run_022_b1_utterance_large_114k_v3_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "python.*pipeline.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur OVH :" >&2
    pgrep -af "python.*pipeline.py (train|run)" >&2 || true
    exit 2
  fi
  echo "OK: aucun pipeline.py train/run Python actif."
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

echo "=== $(date -Is) Pré-vol speechLLM L-114k v3 (OVH, run_022) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_114k_v3.yaml \
  --run-id run_022_speechllm_b1_utterance_large_114k_v3 \
  --dry-run

echo "=== $(date -Is) Délégation → ${SPEECHLLM_SCRIPT} ==="
exec bash "$SPEECHLLM_SCRIPT"
