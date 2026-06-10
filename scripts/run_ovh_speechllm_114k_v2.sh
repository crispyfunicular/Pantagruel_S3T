#!/usr/bin/env bash
# OVH — speechLLM Pantagruel-L-114k v2 (run_017).
#
# Suite à run_013 (15,92 / 15,24 test) et run_016 ST L-114k v2 terminés sur cette VM.
# Pistes variantes.md §2 : encodeur 114k gelé + max_new_tokens 128 + early stop
# (calqué sur run_014 speechLLM L-14k v2, Modyco).
# Durée : ~4–6 h GPU (early stop possible avant 20k).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e ssh ./ ubuntu@145.239.52.158:~/S3T/
#   ssh ubuntu@145.239.52.158 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_ovh_speechllm_114k_v2.sh > logs/run_017_ovh_chain_wrapper.log 2>&1 &'
#
# Sur OVH :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_ovh_speechllm_114k_v2.sh > logs/run_017_ovh_chain_wrapper.log 2>&1 &
#   tail -f logs/run_017_speechllm_b1_utterance_large_114k_v2_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_ovh_speechllm_114k_v2.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPEECHLLM_SCRIPT="${ROOT}/2_speechLLM/scripts/run_017_b1_utterance_large_114k_v2_nohup.sh"

if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${ROOT}/.venv/bin/activate"
fi

require_gpu_free() {
  # Ne matcher que les processus Python du pipeline (pas les commandes ssh/bash).
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
      sed -n '1,24p' "$0" | tail -n +2
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

echo "=== $(date -Is) Pré-vol speechLLM L-114k v2 (OVH, run_017) ==="
python 2_speechLLM/pipeline.py train \
  --config 2_speechLLM/configs/fr-en/b1_utterance_large_114k_v2.yaml \
  --run-id run_017_speechllm_b1_utterance_large_114k_v2 \
  --dry-run

echo "=== $(date -Is) Délégation → ${SPEECHLLM_SCRIPT} ==="
exec bash "$SPEECHLLM_SCRIPT"
