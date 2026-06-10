#!/usr/bin/env bash
# Modyco — speechLLM B1 Pantagruel-L-14k v2 (retry run_012).
#
# Budget utilisateur : ≤ 12 h GPU. Durée attendue ~1,5–3 h (early stop possible avant 20k).
# Pas de doublon OVH : run_013 (L-114k) reste sur la VM OVH.
#
# Depuis le poste local (nohup sur la tour) :
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_speechllm_14k_v2.sh \
#     > logs/run_014_speechllm_14k_v2_chain_wrapper.log 2>&1 &'
#
# Lancement direct sur Modyco :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_speechllm_14k_v2.sh \
#     > logs/run_014_speechllm_14k_v2_chain_wrapper.log 2>&1 &
#   tail -f logs/run_014_speechllm_b1_utterance_large_14k_v2_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_speechllm_14k_v2.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPEECHLLM_SCRIPT="${ROOT}/2_speechLLM/scripts/run_014_b1_utterance_large_14k_v2_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  if pgrep -af "pipeline.py (train|run)" >/dev/null 2>&1; then
    echo "ERROR: entraînement GPU encore actif sur Modyco :" >&2
    pgrep -af "pipeline.py (train|run)" >&2 || true
    echo "  Attendre la fin ou relancer avec --force." >&2
    exit 2
  fi
  echo "OK: aucun pipeline.py train/run actif."
}

usage() {
  sed -n '1,22p' "$0" | tail -n +2
}

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help|help) usage; exit 0 ;;
    *)
      echo "Argument inconnu: ${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$FORCE" != "1" ]]; then
  require_gpu_free
fi

if [[ ! -x "$SPEECHLLM_SCRIPT" ]]; then
  chmod +x "$SPEECHLLM_SCRIPT"
fi

echo "=== $(date -Is) Pré-vol speechLLM L-14k v2 (Modyco, run_014) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_14k_v2.yaml \
  --run-id run_014_speechllm_b1_utterance_large_14k_v2 \
  --dry-run

echo "=== $(date -Is) Délégation → ${SPEECHLLM_SCRIPT} ==="
exec bash "$SPEECHLLM_SCRIPT"
