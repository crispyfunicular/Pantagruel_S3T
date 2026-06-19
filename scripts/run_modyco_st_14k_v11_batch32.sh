#!/usr/bin/env bash
# Modyco — ST Pantagruel-L-14k v11 batch effectif 32 (run_046, piste B).
#
# Durée attendue : ~8–10 h GPU. À lancer après run_037 (ou Piste D rééval).
#
# Depuis le poste local :
#   rsync -az --exclude '.venv' --exclude 'runs/' --exclude 'datasets/' \
#     -e "ssh -i ~/.ssh/id_ed25519" ./ mpellissier@10.8.0.2:~/S3T/
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_st_14k_v11_batch32.sh \
#     > logs/run_046_st_14k_v11_batch32_chain_wrapper.log 2>&1 &'

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_046_baseline_utterance_large_14k_v11_batch32_nohup.sh"

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
      sed -n '1,12p' "$0" | tail -n +2
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

echo "=== $(date -Is) Pré-vol ST L-14k v11 batch-32 (Modyco, run_046) ==="
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_large_14k_v11_batch32.yaml \
  --run-id run_046_transformer_baseline_utterance_large_14k_v11_batch32 \
  --dry-run

echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
