#!/usr/bin/env bash
# Modyco — ST Pantagruel-L-114k v3 (run_019).
#
# Retry run_016 (19,63 test OVH) avec early stop patience 4 et eval dev complet.
# Budget utilisateur : ≤ 10 h GPU. Durée attendue ~9–10 h (comme run_016).
# Pas de doublon OVH : run_017 speechLLM L-114k v2 en cours sur la VM OVH.
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e "ssh -i ~/.ssh/id_ed25519" ./ mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_st_114k_v3.sh \
#     > logs/run_019_st_114k_v3_chain_wrapper.log 2>&1 &'
#
# Sur Modyco :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_st_114k_v3.sh > logs/run_019_st_114k_v3_chain_wrapper.log 2>&1 &
#   tail -f logs/run_019_transformer_baseline_utterance_large_114k_v3_spm_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_st_114k_v3.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_019_baseline_utterance_large_114k_v3_nohup.sh"

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

chmod +x "$ST_SCRIPT"

echo "=== $(date -Is) Pré-vol ST L-114k v3 (Modyco, run_019) ==="
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_large_114k_v3.yaml \
  --run-id run_019_transformer_baseline_utterance_large_114k_v3 \
  --dry-run

echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
