#!/usr/bin/env bash
# Modyco — ST Pantagruel-B-1k v5 SpecAugment (run_035).
#
# Budget : ≤ 3,5 h GPU. Durée attendue ~2–3 h (encodeur Base, plus léger que L-14k).
# Cible papier Table 8 : ~17,5 BLEU test (run_004 v2 sans SpecAugment : 16,68).
#
# Depuis le poste local :
#   ./scripts/tour.sh ssh 'cd ~/S3T && source .venv/bin/activate && mkdir -p logs && \
#     nohup bash scripts/run_modyco_st_b1k_v5.sh \
#     > logs/run_035_st_b1k_v5_chain_wrapper.log 2>&1 &'
#
# Forcer malgré un job GPU actif (déconseillé) :
#   bash scripts/run_modyco_st_b1k_v5.sh --force

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ST_SCRIPT="${ROOT}/1_Transformer/scripts/run_035_baseline_utterance_b1k_v5_nohup.sh"

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
      sed -n '1,16p' "$0" | tail -n +2
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

echo "=== $(date -Is) Pré-vol ST B-1k v5 SpecAugment (Modyco, run_035) ==="
python 1_Transformer/pipeline.py train \
  --config 1_Transformer/configs/fr-en/base_utterance_b1k_v5.yaml \
  --run-id run_035_transformer_baseline_utterance_b1k_v5 \
  --dry-run

echo "=== $(date -Is) Délégation → ${ST_SCRIPT} ==="
exec bash "$ST_SCRIPT"
