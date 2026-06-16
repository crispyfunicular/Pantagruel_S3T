#!/usr/bin/env bash
# Modyco — réplication ST Pantagruel-L-14k v5 (run_043, validation run_026 @ 26,12 BLEU).
#
# Config identique à run_026 ; budget ~7–8 h GPU (early stop @ ~50–55k updates).
# Ne pas lancer en parallèle d'un autre job GPU sur Modyco (serveur partagé).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e "ssh -i ~/.ssh/id_ed25519" ./ mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_st_14k_v5_replicate.sh \
#     > logs/run_043_st_14k_v5_replicate_chain_wrapper.log 2>&1 &'
#
# Sur Modyco :
#   cd ~/S3T && source .venv/bin/activate
#   mkdir -p logs
#   nohup bash scripts/run_modyco_st_14k_v5_replicate.sh \
#     > logs/run_043_st_14k_v5_replicate_chain_wrapper.log 2>&1 &
#   tail -f logs/run_043_transformer_baseline_utterance_large_14k_v5_replicate_spm_train_eval.log
#
# Forcer malgré un job GPU actif (déconseillé sur serveur partagé) :
#   bash scripts/run_modyco_st_14k_v5_replicate.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_043_baseline_utterance_large_14k_v5_replicate_nohup.sh"

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

echo "=== $(date -Is) Pré-vol ST L-14k v5 replicate (Modyco, run_043 — validation run_026) ==="
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_large_14k_v5_replicate.yaml \
  --run-id run_043_transformer_baseline_utterance_large_14k_v5_replicate \
  --dry-run

echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
