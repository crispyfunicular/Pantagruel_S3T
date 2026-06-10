#!/usr/bin/env bash
# OVH — ST Pantagruel-L-114k v2 (run_016).
#
# Suite aux runs speechLLM run_012/013 terminés sur cette VM.
# Pistes variantes.md §1 : encodeur 114k + correctifs run_014 (gel 5k, LR 1e-4, early stop).
# Durée : ~4–12 h GPU (early stop possible).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e ssh ./ ubuntu@145.239.52.158:~/S3T/
#   ssh ubuntu@145.239.52.158 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_ovh_st_114k_v2.sh > logs/run_016_ovh_chain_wrapper.log 2>&1 &'
#
# Sur OVH :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_ovh_st_114k_v2.sh > logs/run_016_ovh_chain_wrapper.log 2>&1 &
#   tail -f logs/run_016_transformer_baseline_utterance_large_114k_v2_spm_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_ovh_st_114k_v2.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_016_baseline_utterance_large_114k_v2_nohup.sh"

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

chmod +x "$ST_SCRIPT"

echo "=== $(date -Is) Pré-vol ST L-114k v2 (OVH, run_016) ==="
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_large_114k_v2.yaml \
  --run-id run_016_transformer_baseline_utterance_large_114k_v2 \
  --dry-run

echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
