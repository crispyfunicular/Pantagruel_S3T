#!/usr/bin/env bash
# Modyco — finir run_036 ST L-14k v9 warmup 10k (reprise ou overwrite).
#
# Depuis le poste local :
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_st_14k_v9_warmup10k_finish.sh \
#     > logs/run_036_st_v9_warmup10k_finish_chain_wrapper.log 2>&1 &'
#
# Variables optionnelles (sur Modyco) :
#   RESUME=1     — reprendre depuis last.pt / best.pt (défaut)
#   OVERWRITE=1  — repartir de zéro (supprime les checkpoints)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_036_baseline_utterance_large_14k_v9_warmup10k_finish_nohup.sh"

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

chmod +x "$ST_SCRIPT"

RESUME="${RESUME:-1}"
OVERWRITE="${OVERWRITE:-0}"

echo "=== $(date -Is) Pré-vol ST L-14k v9 warmup 10k finish (Modyco, run_036) ==="
echo "  RESUME=${RESUME} OVERWRITE=${OVERWRITE}"
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_large_14k_v9_warmup10k.yaml \
  --run-id run_036_transformer_baseline_utterance_large_14k_v9_warmup10k \
  --dry-run \
  $( [[ "$RESUME" == "1" ]] && echo --resume ) \
  $( [[ "$OVERWRITE" == "1" ]] && echo --overwrite )

echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec env RESUME="${RESUME}" OVERWRITE="${OVERWRITE}" bash "$ST_SCRIPT"
